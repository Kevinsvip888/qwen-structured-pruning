#!/usr/bin/env python3
"""Export/reload one frozen BF16 pruned Qwen3 checkpoint into a new directory."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collect_channel_importance import (verify_inputs, verify_saved, POINTER, gpu_pids, gpu_identity,
                                        sha as file_sha)
from prepare_pruning_study import tensor_inventory

METHODS = ('RandomChannels', 'ActivationEnergy')
MLP_PATTERN = re.compile(r'^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'vocab.json', 'merges.txt', 'tokenizer.model', 'chat_template.jinja')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def value_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def save(path, value):
    path.write_bytes(encoded(value))


def normalized_config(config):
    # Loading paths, library version and dtype spelling are serialization metadata.
    # Runtime parameter dtype is checked separately and must remain BF16.
    return {k: v for k, v in config.items()
            if not k.startswith('_') and k not in ('transformers_version', 'dtype', 'torch_dtype')}


def expected_tensor_shape(name, shape, keep):
    match = MLP_PATTERN.fullmatch(name)
    if match:
        return [keep, shape[1]] if match[2] in ('gate_proj','up_proj') else [shape[0],keep]
    return list(shape)


def load_info_ok(info):
    require(not info.get('unexpected_keys') and not info.get('mismatched_keys') and not info.get('error_msgs')
            and set(info.get('missing_keys', [])) <= {'lm_head.weight'}, 'Checkpoint did not load cleanly: ' + str(info))


def validate_surgery(p, importance):
    pointer_path = ROOT / 'data/surgery_validation_v1/latest.json'
    pointer = read(pointer_path)
    receipt_path = ROOT / pointer['receipt']
    require(receipt_path.resolve().is_relative_to(ROOT.resolve()) and file_sha(receipt_path) == pointer['sha256'],
            'Surgery validation receipt changed')
    receipt = read(receipt_path)
    importance_pointer = read(POINTER)
    require(receipt['protocol_sha256'] == file_sha(ROOT/'data/protocol_v1.json')
            and receipt['importance_manifest_sha256'] == importance_pointer['manifest_sha256']
            and receipt['gpu_identity'] == p['gpu_identity'] and receipt['source_checkpoint_files_unchanged'],
            'Surgery validation identity differs')
    require(len(receipt['toy_fp64_checks']) == 42 and len(receipt['real_fp64_checks']) == 144
            and len(receipt['native_bf16_smokes']) == 144 and receipt['exact_weight_copy_checks'] == 144
            and len(receipt['invalid_inputs_rejected']) == 13, 'Surgery validation incomplete')
    for relative, digest in receipt['file_sha256'].items():
        require(file_sha(ROOT/relative) == digest, 'Validated surgery source changed: ' + relative)
    require(receipt['file_sha256']['pruning_lab/surgery.py'] == file_sha(ROOT/'pruning_lab/surgery.py'),
            'Missing validated surgery helper')
    return receipt_path


def checkpoint_weight_map(folder):
    from safetensors import safe_open
    index = folder/'model.safetensors.index.json'
    if index.exists():
        return read(index)['weight_map']
    mapping = {}
    for shard in folder.glob('*.safetensors'):
        with safe_open(shard, framework='pt', device='cpu') as handle:
            for name in handle.keys():
                require(name not in mapping, 'Duplicate exported tensor')
                mapping[name] = shard.name
    return mapping


def audit_disk_tensors(source, target, retained, width):
    """Every exported tensor must equal its source tensor or the prescribed slice."""
    import torch
    from safetensors import safe_open
    source_config, target_config = read(source/'config.json'), read(target/'config.json')
    original_inventory, _, original_count = tensor_inventory(source, source_config)
    inventory, files, count = tensor_inventory(target, target_config)
    require(set(inventory) == set(original_inventory), 'Export tensor names/alias canonicalization differ')
    source_map, target_map = checkpoint_weight_map(source), checkpoint_weight_map(target)
    require(set(source_map) == set(target_map) == set(inventory), 'Checkpoint index coverage differs')
    checks, sliced = 0, 0
    indices = {layer: torch.tensor(ids, dtype=torch.long) for layer, ids in retained.items()}
    for name in sorted(inventory):
        require(inventory[name]['shape'] == expected_tensor_shape(name, original_inventory[name]['shape'], width),
                'Saved shape differs: ' + name)
        with safe_open(source/source_map[name], framework='pt', device='cpu') as h:
            original = h.get_tensor(name)
        with safe_open(target/target_map[name], framework='pt', device='cpu') as h:
            saved = h.get_tensor(name)
        match = MLP_PATTERN.fullmatch(name)
        if match:
            dim = 0 if match[2] in ('gate_proj','up_proj') else 1
            expected = original.index_select(dim, indices[match[1]])
            sliced += 1
        else:
            expected = original
        require(saved.dtype == torch.bfloat16 and torch.equal(saved, expected), 'Saved weight values differ: ' + name)
        checks += 1
        del original,saved,expected
        if checks == 1 or checks % 64 == 0:
            print(f'  Exact saved-weight audit: {checks}/{len(inventory)} tensors', flush=True)
    require(sliced == 108, 'Expected three sliced projections in all 36 layers')
    return {'checked_tensors':checks, 'sliced_mlp_tensors':sliced,
            'unchanged_tensors':checks-sliced, 'original_tensor_elements':original_count,
            'exported_tensor_elements':count,
            'safetensors_file_bytes':sum(x.stat().st_size for x in files)}


def check_architecture(model, p, parameter_count, target_width):
    import torch
    require(model.config.intermediate_size == target_width and len(model.model.layers) == 36
            and model.config.hidden_size == 2560 and model.config.tie_word_embeddings, 'Runtime architecture differs')
    require(model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr(),
            'Embedding/LM-head tie changed')
    require(sum(x.numel() for x in model.parameters()) == parameter_count, 'Runtime unique parameter count differs')
    require(all(x.dtype == torch.bfloat16 and not x.requires_grad and not x.is_meta for x in model.parameters()),
            'Expected frozen BF16 weights')
    for block in model.model.layers:
        m = block.mlp
        require(m.intermediate_size == m.config.intermediate_size == target_width
                and tuple(m.gate_proj.weight.shape) == tuple(m.up_proj.weight.shape) == (target_width,2560)
                and tuple(m.down_proj.weight.shape) == (2560,target_width), 'Runtime MLP width differs')


def probe_model(model, probes):
    import torch
    results = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for item in probes:
            ids = torch.tensor([item['input_ids']], device='cuda', dtype=torch.long)
            result = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                 output_hidden_states=False, output_attentions=False, return_dict=True)
            hidden = result.last_hidden_state[:, -1:, :]
            logits = model.lm_head(hidden)
            require(torch.isfinite(hidden).all().item() and torch.isfinite(logits).all().item(), 'Nonfinite reload probe')
            results.append({'id': item['id'], 'input_ids': item['input_ids'],
                            'last_hidden_state':hidden.cpu(), 'last_token_logits':logits.cpu()})
    return results


def verify_existing(destination, identity):
    m = read(destination/'pruning_manifest.json')
    require(m['identity'] == identity and m['reload_checks']['probe_hidden_states_exact']
            and m['reload_checks']['probe_logits_exact'] and m['source_files_unchanged'],
            'Existing checkpoint identity/checks differ; preserve it')
    for relative, info in m['export_files'].items():
        path = destination/relative
        require(path.stat().st_size == info['bytes'] and file_sha(path) == info['sha256'], 'Export file changed: ' + relative)
    config = read(destination/'config.json')
    require(config['intermediate_size'] == 9216 and not config.get('quantization_config'), 'Existing export config differs')
    print('PASS: existing completed checkpoint hashes and recorded reload checks verified; no new export or GPU probe run.')
    print('Checkpoint:',destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',required=True,choices=METHODS)
    args=parser.parse_args()
    p,rows=verify_inputs()
    importance=verify_saved(POINTER,p)
    surgery_receipt=validate_surgery(p,importance)
    from pruning_lab.surgery import prune_mlp
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source=Path(p['source_model']['local_dir'])
    selections_path=ROOT/importance['selection_file']
    retained=read(selections_path)['methods'][args.method]
    require(set(retained)=={str(i) for i in range(36)},'Selected layer coverage differs')
    for ids in retained.values():
        require(len(ids)==9216 and ids==sorted(set(ids)) and 0<=ids[0]<=ids[-1]<9728,'Invalid retained indices')
    identity={'method':args.method,'target_intermediate_size':9216,
              'protocol_sha256':file_sha(ROOT/'data/protocol_v1.json'),
              'importance_manifest_sha256':read(POINTER)['manifest_sha256'],
              'surgery_validation_sha256':file_sha(surgery_receipt),
              'retained_indices_sha256':value_sha(retained),
              'exporter_script_sha256':file_sha(Path(__file__)),
              'surgery_helper_sha256':file_sha(ROOT/'pruning_lab/surgery.py'),
              'dtype':'bfloat16','fine_tuning':False,'quantization':False}
    destination=ROOT/'models'/f'Qwen3-4B-Instruct-2507-MLP9216-{args.method}'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        verify_existing(destination,identity)
        return
    require(not gpu_pids(),'Stop other GPU jobs before exporting and reloading')
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA/BF16 unavailable')
    free,_=torch.cuda.mem_get_info()
    require(free>=16*2**30,'Need at least 16 GiB free GPU memory')
    require(shutil.disk_usage(destination.parent).free>=12*2**30,'Need at least 12 GiB free disk space per export')
    torch.set_num_threads(min(4,len(os.sched_getaffinity(0))));torch.manual_seed(42)
    temporary=Path(tempfile.mkdtemp(prefix=destination.name+'-building-',dir=destination.parent))
    print('Temporary build directory:',temporary,flush=True)
    started=time.perf_counter()
    print('Loading a fresh original BF16 model on CPU...',flush=True)
    model,loading=AutoModelForCausalLM.from_pretrained(source,dtype=torch.bfloat16,
                 attn_implementation='eager',local_files_only=True,output_loading_info=True)
    load_info_ok(loading);model.eval();model.requires_grad_(False)
    original_parameters=sum(x.numel() for x in model.parameters())
    require(original_parameters==p['source_serialized_tensor_elements'],'Original parameter count differs')
    original_config=copy.deepcopy(model.config.to_dict())
    require(normalized_config(original_config)['intermediate_size']==9728,'Expected original source width')
    # Use the already-available GPU for slicing, avoiding an extra full set of
    # compact MLP weights alongside the original checkpoint in host memory.
    model=model.to('cuda')
    gc.collect()
    for layer in range(36):
        old_mlp=model.model.layers[layer].mlp
        new_mlp=prune_mlp(old_mlp,retained[str(layer)])
        model.model.layers[layer].mlp=new_mlp
        if layer==0 or (layer+1)%8==0 or layer==35:
            print(f'Pruned layer {layer+1}/36: 9728 -> 9216 channels',flush=True)
    del old_mlp,new_mlp
    # Set the shared/global config only after every original MLP has been replaced.
    model.config.intermediate_size=9216
    model.model.config.intermediate_size=9216
    expected_config=normalized_config(original_config);expected_config['intermediate_size']=9216
    require(normalized_config(model.config.to_dict())==expected_config,'Unexpected semantic configuration change')
    target_parameters=original_parameters-p['removed_mlp_weight_parameters']
    check_architecture(model,p,target_parameters,9216)
    model=model.to('cuda')
    chosen=[]
    for domain,length in (('general',32),('math',64)):
        row=next(x for x in rows if x['domain']==domain)
        chosen.append({'id':row['id'],'input_ids':row['input_ids'][:length]})
    print('Checking two finite pre-export backbone/LM-head probes...',flush=True)
    before=probe_model(model,chosen)
    print('Saving the pruned BF16 checkpoint...',flush=True)
    model.save_pretrained(temporary,safe_serialization=True,max_shard_size='2GB')
    copied=[]
    for name in TOKENIZER_FILES+('generation_config.json',):
        if (source/name).is_file():
            shutil.copy2(source/name,temporary/name);copied.append(name)
    require('tokenizer.json' in copied and 'tokenizer_config.json' in copied,'Missing source tokenizer files')
    del model
    gc.collect();torch.cuda.empty_cache()
    # Every saved non-MLP value must be unchanged; every MLP value must be exactly the prescribed slice.
    audit=audit_disk_tensors(source,temporary,retained,9216)
    require(audit['exported_tensor_elements']==target_parameters
            and audit['original_tensor_elements']-audit['exported_tensor_elements']==141557760,
            'Saved parameter reduction differs')
    require(not read(temporary/'config.json').get('quantization_config'),'Export must remain BF16 dense')
    print('Reloading the exported checkpoint...',flush=True)
    reloaded,loading=AutoModelForCausalLM.from_pretrained(temporary,dtype=torch.bfloat16,
                 attn_implementation='eager',local_files_only=True,output_loading_info=True)
    load_info_ok(loading);reloaded.eval();reloaded.requires_grad_(False)
    check_architecture(reloaded,p,target_parameters,9216)
    require(normalized_config(reloaded.config.to_dict())==expected_config,'Reloaded semantic configuration differs')
    # Check actual reloaded parameters as well as the serialized tensor contents.
    from safetensors import safe_open
    mapping=checkpoint_weight_map(temporary)
    loaded_checks=0
    for name,value in reloaded.named_parameters():
        require(name in mapping,'Unexpected reloaded parameter: '+name)
        with safe_open(temporary/mapping[name],framework='pt',device='cpu') as h:
            expected=h.get_tensor(name)
        require(torch.equal(value.detach().cpu(),expected),'Reloaded parameter differs: '+name)
        loaded_checks+=1
    require(loaded_checks==audit['checked_tensors'],'Reloaded unique parameter coverage differs')
    reloaded=reloaded.to('cuda')
    after=probe_model(reloaded,chosen)
    require(all(torch.equal(a['last_hidden_state'],b['last_hidden_state'])
                and torch.equal(a['last_token_logits'],b['last_token_logits']) for a,b in zip(before,after)),
            'Reloaded probe outputs differ from pre-export outputs; preserve the build directory for review')
    original_tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    saved_tokenizer=AutoTokenizer.from_pretrained(temporary,local_files_only=True)
    require(original_tokenizer.get_vocab()==saved_tokenizer.get_vocab()
            and original_tokenizer.chat_template==saved_tokenizer.chat_template
            and original_tokenizer.all_special_ids==saved_tokenizer.all_special_ids,'Tokenizer behavior changed')
    for text in ('Calculate 17 multiplied by 23.','A temperature sensor records a reading every minute.','你好，世界。'):
        require(original_tokenizer.encode(text,add_special_tokens=False)==saved_tokenizer.encode(text,add_special_tokens=False),
                'Tokenizer probe differs')
    for name in copied:
        require(file_sha(source/name)==file_sha(temporary/name),'Copied tokenizer/generation file differs')
    del reloaded,original_tokenizer,saved_tokenizer
    gc.collect();torch.cuda.empty_cache()
    require(gpu_identity()==p['gpu_identity'],'Physical GPU changed')
    print('Verifying source files remain unchanged and hashing export files...',flush=True)
    for filename,digest in p['source_file_sha256'].items():
        require(file_sha(Path(filename))==digest,'Original checkpoint file changed')
    for relative,digest in read(surgery_receipt)['file_sha256'].items():
        require(file_sha(ROOT/relative)==digest,'Validated source changed during export')
    require(file_sha(Path(__file__))==identity['exporter_script_sha256'],'Exporter changed during the run')
    exported={str(x.relative_to(temporary)):{'bytes':x.stat().st_size,'sha256':file_sha(x)}
              for x in sorted(temporary.rglob('*')) if x.is_file()}
    manifest={'identity':identity,'source_model':p['source_model'],'destination':str(destination),
              'original_unique_parameters':original_parameters,'pruned_unique_parameters':target_parameters,
              'removed_parameters':original_parameters-target_parameters,'retained_indices':retained,
              'disk_tensor_audit':audit,'export_files':exported,'source_files_unchanged':True,
              'reload_checks':{'unique_parameter_tensors_checked':loaded_checks,'tied_embeddings':True,
                               'probe_hidden_states_exact':True,'probe_logits_exact':True,'tokenizer_unchanged':True,
                               'probes':chosen},
              'scope':'BF16 structured channel-pruning export and reload checks only; no quality or speed conclusion.'}
    save(temporary/'pruning_manifest.json',manifest)
    require(not destination.exists(),'Destination appeared during build; no checkpoint will be overwritten')
    temporary.rename(destination)
    receipt_dir=ROOT/'outputs/exports_v1'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt_dir.mkdir(parents=True,exist_ok=False)
    receipt={'checkpoint':str(destination),'method':args.method,'identity':identity,
             'manifest_sha256':file_sha(destination/'pruning_manifest.json'),
             'elapsed_export_and_validation_seconds':time.perf_counter()-started,
             'quality_or_speed_measured':False}
    save(receipt_dir/'export_receipt.json',receipt)
    (receipt_dir/'export_receipt.sha256').write_text(file_sha(receipt_dir/'export_receipt.json')+'\n')
    print('\nMethod:',args.method)
    print('Original unique parameters:',f'{original_parameters:,}')
    print('Pruned unique parameters:',f'{target_parameters:,}')
    print('Removed parameters:',f'{original_parameters-target_parameters:,}')
    print('Exported MLP width: 9,216; dtype: BF16; tied embeddings: preserved')
    print('Exact saved tensor checks:',audit['checked_tensors'],'; reloaded parameter checks:',loaded_checks)
    print('Safetensors size GiB:',f'{audit["safetensors_file_bytes"]/2**30:.4f}')
    print('Checkpoint:',destination)
    print('Receipt:',receipt_dir/'export_receipt.json')
    print('PASS: prescribed slices, unchanged other weights, tokenizer, tied embeddings and exact reload probes verified.')
    print('Quality and serving speed remain unmeasured for this pruned checkpoint.')


if __name__=='__main__':
    main()
