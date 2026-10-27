#!/usr/bin/env python3
"""Check compact MLPs against masked dense MLPs before any model export."""
from datetime import datetime, timezone
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collect_channel_importance import verify_inputs, verify_saved, POINTER, gpu_pids, gpu_identity


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    p, _ = verify_inputs()
    importance = verify_saved(POINTER, p)
    require(not gpu_pids(), 'Stop other GPU jobs before validation')
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    from pruning_lab.surgery import prune_mlp

    spec_path = ROOT / 'specs/surgery_validation_v1.json'
    spec = read(spec_path)
    require(spec['version'] == 1 and spec['fp64_tolerances'] == {'rtol': 1e-10, 'atol': 1e-10}
            and spec['real_layers'] == 36 and spec['methods'] == ['RandomChannels','ActivationEnergy']
            and spec['input_shapes_per_real_layer'] == [[1,2560],[1,4,2560]], 'Validation contract changed')
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA/BF16 unavailable')
    free, _ = torch.cuda.mem_get_info()
    require(free >= 4 * 2**30, 'Need at least 4 GiB free GPU memory for layer validation')
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    torch.manual_seed(spec['real_input_seed'])
    require(sha(Path(inspect.getfile(Qwen3MLP))) == importance['modeling_source_sha256'], 'Model implementation changed')
    selections = read(ROOT / importance['selection_file'])['methods']
    pointer = read(POINTER)
    manifest_path = ROOT / pointer['manifest']
    source_code = {str(x.relative_to(ROOT)): sha(x) for x in (
        Path(__file__).resolve(), ROOT/'pruning_lab/surgery.py', ROOT/'pruning_lab/__init__.py',
        ROOT/'scripts/collect_channel_importance.py', ROOT/'scripts/prepare_pruning_study.py', spec_path)}

    def snapshot(module):
        return {name: value.detach().clone() for name, value in module.named_parameters()}

    def check_copy(original, compact, retained, old_values):
        h, k = original.hidden_size, len(retained)
        require(compact is not original and compact.config is not original.config
                and compact.intermediate_size == compact.config.intermediate_size == k
                and original.intermediate_size == original.config.intermediate_size,
                'MLP configuration changed or was shared')
        indices = torch.tensor(retained, device=original.gate_proj.weight.device, dtype=torch.long)
        expected_shapes = {'gate_proj':(k,h), 'up_proj':(k,h), 'down_proj':(h,k)}
        source_ptrs = {x.untyped_storage().data_ptr() for x in original.parameters()}
        compact_ptrs = []
        for name, dim in (('gate_proj',0),('up_proj',0),('down_proj',1)):
            old, new = getattr(original,name), getattr(compact,name)
            require(tuple(new.weight.shape) == expected_shapes[name]
                    and (new.out_features,new.in_features) == expected_shapes[name]
                    and new.weight.dtype == old.weight.dtype and new.weight.device == old.weight.device
                    and torch.equal(new.weight,old.weight.index_select(dim,indices)), 'Weight slice differs: '+name)
            compact_ptrs.append(new.weight.untyped_storage().data_ptr())
        require(len(set(compact_ptrs)) == 3 and not source_ptrs.intersection(compact_ptrs), 'Weight storage aliases the source')
        require(all(not x.requires_grad and not x.is_meta for x in compact.parameters()), 'Invalid compact parameter state')
        require(all(torch.equal(dict(original.named_parameters())[n],v) for n,v in old_values.items()), 'Original weights changed')
        expected_removed = 3 * h * (original.intermediate_size-k)
        require(sum(x.numel() for x in original.parameters())-sum(x.numel() for x in compact.parameters()) == expected_removed,
                'Parameter reduction differs')

    def masked_reference(original, inputs, retained):
        z = F.silu(F.linear(inputs, original.gate_proj.weight)) * F.linear(inputs, original.up_proj.weight)
        mask = torch.zeros(original.intermediate_size, device=inputs.device, dtype=inputs.dtype)
        mask[retained] = 1
        return F.linear(z * mask, original.down_proj.weight)

    def compare_fp64(original, compact, retained, inputs, label):
        before = inputs.clone()
        expected = masked_reference(original, inputs, retained)
        actual = compact(inputs)
        torch.testing.assert_close(actual, expected, **spec['fp64_tolerances'])
        require(torch.isfinite(actual).all().item() and torch.equal(inputs,before), 'Nonfinite output or mutated input')
        return {'label':label, 'input_shape':list(inputs.shape),
                'max_absolute_error':float((actual-expected).abs().max().item())}

    # Test the exact same helper on CPU and GPU, including identity/single/tail selections.
    print('Checking toy MLPs on CPU and GPU, including identity and tail selections...', flush=True)
    toys = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for device in ('cpu','cuda'):
            for hidden,width in ((4,7),(8,12),(17,33)):
                cfg=Qwen3Config(hidden_size=hidden, intermediate_size=width, num_hidden_layers=1,
                               num_attention_heads=1, num_key_value_heads=1, head_dim=4,
                               vocab_size=32, hidden_act='silu')
                original=Qwen3MLP(cfg).to(device=device,dtype=torch.float64).eval()
                old=snapshot(original)
                for retained in ([width-1], list(range(0,width,2)), list(range(width))):
                    config_before=copy.deepcopy(original.config.to_dict())
                    compact=prune_mlp(original,retained)
                    check_copy(original,compact,retained,old)
                    for shape in ((1,hidden),(1,3,hidden)):
                        x=torch.randn(shape,device=device,dtype=torch.float64)
                        toys.append(compare_fp64(original,compact,retained,x,f'{device}/{hidden}x{width}/keep{len(retained)}'))
                    require(original.config.to_dict()==config_before,'Source configuration mutated')
                # Zero-input oracle boundary.
                compact=prune_mlp(original,[width-1])
                x=torch.zeros(2,hidden,device=device,dtype=torch.float64)
                toys.append(compare_fp64(original,compact,[width-1],x,f'{device}/zero/{hidden}x{width}'))
    require(len(toys)==42,'Toy check count differs')
    print('PASS: 42 toy FP64 masked/compact comparisons.', flush=True)

    cfg=Qwen3Config(hidden_size=4,intermediate_size=7,num_hidden_layers=1,num_attention_heads=1,
                   num_key_value_heads=1,head_dim=4,vocab_size=32,hidden_act='silu')
    original=Qwen3MLP(cfg).double().eval()
    cases=[('empty',[],ValueError),('duplicate',[1,1],ValueError),('unsorted',[2,0],ValueError),
           ('negative',[-1,0],ValueError),('out of range',[0,7],ValueError),('boolean',[True],TypeError),
           ('float',[1.0],TypeError),('string','1',TypeError),('tensor',torch.tensor([1]),TypeError)]
    rejected=[]
    for label,indices,error_type in cases:
        try:prune_mlp(original,indices)
        except error_type:rejected.append(label)
        else:raise RuntimeError('Invalid selection accepted: '+label)
    for label,mutate in (
        ('projection bias',lambda m:setattr(m,'gate_proj',torch.nn.Linear(4,7,bias=True).double())),
        ('mixed dtype',lambda m:setattr(m.up_proj,'weight',torch.nn.Parameter(m.up_proj.weight.float()))),
        ('wrong projection shape',lambda m:setattr(m,'down_proj',torch.nn.Linear(6,4,bias=False).double())),
        ('unsupported activation',lambda m:setattr(m.config,'hidden_act','relu'))):
        invalid=copy.deepcopy(original);mutate(invalid)
        try:prune_mlp(invalid,[0,2])
        except ValueError:rejected.append(label)
        else:raise RuntimeError('Invalid MLP accepted: '+label)
    require(len(rejected)==13,'Invalid-input check count differs')

    # Read one original MLP at a time. No full model is loaded or changed.
    model_dir=Path(p['source_model']['local_dir'])
    config=AutoConfig.from_pretrained(model_dir,local_files_only=True)
    mapping=read(model_dir/'model.safetensors.index.json')['weight_map']
    real, native, copies = [],[],0

    def load_layer(layer,dtype):
        local=copy.deepcopy(config)
        with torch.device('meta'):
            mlp=Qwen3MLP(local)
        for projection in ('gate_proj','up_proj','down_proj'):
            name=f'model.layers.{layer}.mlp.{projection}.weight'
            with safe_open(model_dir/mapping[name],framework='pt',device='cpu') as handle:
                value=handle.get_tensor(name)
            value=value.to(device='cuda',dtype=dtype).contiguous()
            getattr(mlp,projection).weight=torch.nn.Parameter(value,requires_grad=False)
        return mlp.eval()

    with torch.inference_mode(), torch.autocast(device_type='cuda',enabled=False):
        for layer in range(36):
            original64=load_layer(layer,torch.float64)
            original16=load_layer(layer,torch.bfloat16)
            old64,old16=snapshot(original64),snapshot(original16)
            original_config=copy.deepcopy(original64.config.to_dict())
            for method in spec['methods']:
                retained=selections[method][str(layer)]
                require(len(retained)==9216,'Selected width differs')
                compact64=prune_mlp(original64,retained)
                compact16=prune_mlp(original16,retained)
                check_copy(original64,compact64,retained,old64)
                check_copy(original16,compact16,retained,old16)
                copies+=2
                for shape in spec['input_shapes_per_real_layer']:
                    x=torch.randn(tuple(shape),device='cuda',dtype=torch.float64)
                    real.append(compare_fp64(original64,compact64,retained,x,f'layer{layer}/{method}'))
                    x16=x.to(torch.bfloat16);before=x16.clone()
                    y16=compact16(x16)
                    require(tuple(y16.shape)==tuple(shape) and y16.dtype==torch.bfloat16
                            and torch.isfinite(y16).all().item() and torch.equal(x16,before),
                            'Native BF16 smoke failed')
                    native.append({'layer':layer,'method':method,'input_shape':shape,'finite_output':True})
                require(original64.config.to_dict()==original_config and original16.config.to_dict()==original_config,
                        'Source layer configuration changed')
                del compact64,compact16
            del original64,original16,old64,old16
            print(f'PASS layer {layer+1}/36: both selections, exact slices, FP64 masked equivalence and native BF16 smoke.',flush=True)
    require(len(real)==len(native)==copies==144,'Real-layer test counts differ')
    for relative,digest in source_code.items():
        require(sha(ROOT/relative)==digest,'Validation code changed')
    require(sha(manifest_path)==pointer['manifest_sha256'] and gpu_identity()==p['gpu_identity'], 'Importance identity or GPU changed')
    # Recheck original files using streaming hashes; keep the weight files read-only.
    from collect_channel_importance import sha as streaming_sha
    for filename,digest in p['source_file_sha256'].items():
        require(streaming_sha(Path(filename))==digest,'Source checkpoint file changed')
    folder=ROOT/'outputs/surgery_validation_v1'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    result={'stage':'Layer surgery validation; no pruned checkpoint exported',
            'protocol_sha256':sha(ROOT/'data/protocol_v1.json'),'importance_manifest_sha256':sha(manifest_path),
            'file_sha256':source_code,'surgery_spec_sha256':sha(spec_path),'gpu_identity':p['gpu_identity'],
            'toy_fp64_checks':toys,'real_fp64_checks':real,'native_bf16_smokes':native,
            'exact_weight_copy_checks':copies,'invalid_inputs_rejected':rejected,
            'source_checkpoint_files_unchanged':True,'performance_or_quality_measured':False,
            'scope':spec['scope']}
    receipt=folder/'surgery_validation.json';save(receipt,result)
    pointer_path=ROOT/'data/surgery_validation_v1/latest.json';pointer_path.parent.mkdir(parents=True,exist_ok=True)
    new_pointer={'receipt':str(receipt.relative_to(ROOT)),'sha256':sha(receipt)}
    if pointer_path.exists():
        old=read(pointer_path);old_receipt=ROOT/old['receipt']
        require(sha(old_receipt)==old['sha256'],'Previous validation receipt changed')
    temporary=pointer_path.with_suffix('.tmp');save(temporary,new_pointer);temporary.replace(pointer_path)
    print('\nPASS: 42 toy FP64 checks, 144 real-layer FP64 checks, 144 native BF16 smokes and 13 rejection checks.')
    print('PASS: 144 exact projection-copy checks; original weights/configurations preserved.')
    print('Maximum real FP64 absolute error:',max(x['max_absolute_error'] for x in real))
    print('Saved:',receipt)
    print('No pruned checkpoint exported. This validates surgery, not model quality or speed.')


if __name__=='__main__':
    main()
