#!/usr/bin/env python3
"""Reconcile parameter counts, allowing for the local decoder's renamed head."""
import json
from pathlib import Path
from prepare_comparison import ROOT, write_json


def main():
    upstream = json.loads((ROOT / 'profiles/deim_s_phone.json').read_text())
    local = json.loads((ROOT / 'profiles/local_plain_phone.json').read_text())
    old = upstream['parameter_schema']
    new = {k.replace('.mi_heads.0.', '.'): v for k,v in local['parameter_schema'].items()}
    assert len(new) == len(local['parameter_schema'])
    extras = {k:v for k,v in new.items() if k not in old}
    missing = {k:v for k,v in old.items() if k not in new}
    changed = {k: {'upstream': old[k], 'local': new[k]} for k in old.keys() & new.keys()
               if old[k]['shape'] != new[k]['shape']}
    groups = {
        'decoder_QueryFusion': sum(v['numel'] for k,v in extras.items() if '.q_fusion.' in k),
        'encoder_DySample': sum(v['numel'] for k,v in extras.items() if k.startswith('encoder.upsample_ops.')),
        'encoder_learned_position_embeddings': sum(v['numel'] for k,v in extras.items() if k in ['encoder.row_embed.weight', 'encoder.col_embed.weight']),
        'frozen_graph_gates': sum(v['numel'] for k,v in extras.items() if k.endswith('.gnn_gate')),
    }
    delta = local['params_training_total'] - upstream['params_training_total']
    assert not missing and not changed
    assert sum(groups.values()) == sum(v['numel'] for v in extras.values()) == delta
    evidence = {
        'scope': '52-class instantiated parameter schemas and targeted forward-code inspection; not proof of full numerical equivalence of all common components',
        'upstream_training_params': upstream['params_training_total'],
        'local_plain_training_params': local['params_training_total'],
        'difference': delta, 'difference_components': groups,
        'upstream_deploy_params': upstream['params_deploy_total'],
        'local_plain_deploy_params': local['params_deploy_total'],
        'canonicalization': 'local decoder mi_heads.0 maps to the corresponding upstream decoder sublayer',
        'extra_parameters': extras,
        'code_evidence': [
            {'file': '/home/wuxi/Dent-FINE/engine/deim/dfine_decoder.py', 'line': 1492, 'finding': 'QueryFusion forward is called unconditionally after the inquiry head(s).'},
            {'file': '/home/wuxi/Dent-FINE/engine/deim/hybrid_encoder.py', 'line': 4238, 'finding': 'FPN top-down upsampling uses DySample.'},
            {'file': '/home/wuxi/Dent-FINE/engine/deim/hybrid_encoder.py', 'line': 4212, 'finding': 'Learned row/column position embeddings are used when initialized; fixed evaluation size initializes them.'},
        ],
        'conclusion': 'The existing local Plain DEIM is a project-internal ablation control, not the pristine official DEIM architecture. Existing paired differences isolate additional recipe changes on that local base, not all changes relative to official DEIM.',
        'historical_10_2M': 'Official 52-class deployed parameters round to 10.2 M, but the source of a historical rounded table value cannot be proved by numerical agreement alone.',
    }
    write_json(ROOT / 'architecture_audit.json', evidence)
    lines = [
        '# 原始 DEIM 与项目内 Plain DEIM：参数与代码核对', '',
        '本次为独立的官方来源比较。此前 Plain DEIM 应理解为项目内消融对照，不能等同于未经修改的官方 DEIM。', '',
        '| 52 类配置 | 训练结构总参数 | deploy 后总参数 |',
        '|---|---:|---:|',
        f"| 官方 DEIM-D-FINE-S | {upstream['params_training_total']:,} | {upstream['params_deploy_total']:,} |",
        f"| 项目内 Plain DEIM | {local['params_training_total']:,} | {local['params_deploy_total']:,} |", '',
        f'训练结构总参数相差 {delta:,}，按规范化后的参数名可分解为：', '',
    ]
    lines += [f'- {k}: {v:,}' for k,v in groups.items()]
    lines += ['', '代码检查确认 QueryFusion、DySample 和可学习位置编码仍参与旧 Plain 的 forward。关闭本轮模块开关并不会自动移除这些历史基础改动。', '',
              '因此，既有三种子配对差值衡量的是在项目内部基础结构上增加 DualOpt/CMFA/SpatialShape 配置的收益；新官方重训将另行衡量相对官方 DEIM 的总体差值。两类证据均保留，不互相替换。', '',
              '历史 10.2 M 与官方 deploy 后四舍五入数值相容，但仅凭数值相同不能证明历史记录的真实来源。训练结构与 deploy 结构的统计口径也需明确区分。', '',
              '该检查定位了参数差异及三个实际执行路径，不代表所有同名组件已被证明数值等价。完整来源、参数名和代码位置见 architecture_audit.json。', '']
    (ROOT / 'architecture_audit.md').write_text('\n'.join(lines))
    print(json.dumps({'difference': delta, 'components': groups}, indent=2))


if __name__ == '__main__':
    main()
