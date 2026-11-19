# # models/utils_arch.py
import torch

def fit_arch_t(points_xy: torch.Tensor) -> torch.Tensor:
    """给同一颌的若干点估计沿弓的参数 t∈[0,1]；简化用 x 归一化即可，鲁棒且稳定。"""
    x = points_xy[:, 0]
    t = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return t.clamp_(0, 1)

from typing import Dict, List, Optional, Sequence, Union

# --- NEW: 全局覆盖（按“数据集类ID顺序”给出对应的FDI列表） ---
_FDI_LABEL_ORDER_OVERRIDE: Optional[List[int]] = None

def set_fdi_label_order(order: List[int]) -> None:
    """
    设定数据集“类ID(0..N-1) -> FDI”的顺序表（长度=N，元素是FDI编码）。
    之后 fdi_index_map() 会优先使用此顺序。
    """
    global _FDI_LABEL_ORDER_OVERRIDE
    assert isinstance(order, list) and len(order) > 0
    # 允许的FDI（恒牙）：11..18,21..28,31..38,41..48；乳牙：51..55,61..65,71..75,81..85
    allowed = {q*10+t for q in (1,2,3,4,5,6,7,8) for t in range(1, (9 if q<=4 else 6))}
    for v in order:
        if v not in allowed:
            raise ValueError(f"Bad FDI code in order: {v}")
    _FDI_LABEL_ORDER_OVERRIDE = order[:]  # 复制一份

def clear_fdi_label_order() -> None:
    global _FDI_LABEL_ORDER_OVERRIDE
    _FDI_LABEL_ORDER_OVERRIDE = None

def fdi_permanent_label_order() -> List[int]:
    """Dataset class-id order for 32 permanent FDI labels."""
    return [11,12,13,14,15,16,17,18, 21,22,23,24,25,26,27,28,
            31,32,33,34,35,36,37,38, 41,42,43,44,45,46,47,48]

def fdi_mixed_label_order() -> List[int]:
    """Dataset class-id order for 32 permanent + 20 primary FDI labels."""
    return fdi_permanent_label_order() + [
        51,52,53,54,55, 61,62,63,64,65,
        71,72,73,74,75, 81,82,83,84,85,
    ]

def enumeration32_universal_to_fdi_order() -> List[int]:
    """
    DENTEX enumeration32（Universal #1..#32）-> FDI 的类顺序表（按类ID从0开始）：
      1..16: 上颌右后三到上颌左后三 -> FDI: 18..11,21..28
     17..32: 下颌左后三到下颌右后三 -> FDI: 38..31,41..48
    """
    return [18,17,16,15,14,13,12,11, 21,22,23,24,25,26,27,28,
            38,37,36,35,34,33,32,31, 41,42,43,44,45,46,47,48]
# ------------------------------------------------------------------

def default_fdi52_order() -> List[int]:
    """默认手机/口内照数据的 52 类 FDI 顺序：32 恒牙 + 20 乳牙。"""
    return fdi_mixed_label_order()


def resolve_fdi_label_order(
    order: Optional[Union[str, Sequence[int]]],
    num_classes: Optional[int] = None,
) -> Optional[List[int]]:
    """
    将配置里的 fdi_label_order 解析成“类ID -> FDI”的列表。

    支持：
      - None / "default" / "none"：清除覆盖，使用默认 FDI 顺序
      - "dentex_enumeration32" / "enumeration32_universal"：DENTEX Universal #1..#32 到 FDI
      - "fdi52" / "phone_fdi52"：52 类 FDI 顺序
      - 显式整数列表
    """
    if order is None:
        return None
    if isinstance(order, str):
        key = order.strip().lower()
        if key in ("default", "none"):
            return None
        if key in ("dentex_enumeration32", "enumeration32_universal", "dentex32"):
            resolved = enumeration32_universal_to_fdi_order()
        elif key in ("fdi52", "phone_fdi52", "default52"):
            resolved = default_fdi52_order()
        else:
            raise ValueError(f"Unknown fdi_label_order preset: {order}")
    else:
        resolved = [int(v) for v in order]

    if num_classes is not None:
        resolved = resolved[:int(num_classes)]
    return resolved


def configure_fdi_label_order(
    order: Optional[Union[str, Sequence[int]]],
    num_classes: Optional[int] = None,
) -> None:
    resolved = resolve_fdi_label_order(order, num_classes=num_classes)
    if resolved is None:
        clear_fdi_label_order()
    else:
        set_fdi_label_order(resolved)

def fdi_sequence_targets(is_upper: bool, dentition: str = "perm") -> List[int]:
    """
    dentition: "perm"=恒牙, "primary"=乳牙
    上颌(1/2/5/6), 下颌(3/4/7/8)
    返回图像坐标从左到右的牙弓顺序，用于几何连续性监督。
    """
    perm_upper  = [18,17,16,15,14,13,12,11, 21,22,23,24,25,26,27,28]
    perm_lower  = [48,47,46,45,44,43,42,41, 31,32,33,34,35,36,37,38]
    prim_upper  = [55,54,53,52,51, 61,62,63,64,65]
    prim_lower  = [85,84,83,82,81, 71,72,73,74,75]
    if dentition == "primary":
        return prim_upper if is_upper else prim_lower
    else:
        return perm_upper if is_upper else perm_lower

def fdi_index_map(num_classes: int = 52) -> Dict[int, int]:
    """
    返回 {FDI -> 类ID} 的映射。
    - 若已通过 set_fdi_label_order() 提供了覆盖顺序，则优先使用它（前 num_classes 个）。
    - 否则默认顺序：32 恒牙 + 20 乳牙（按 1..4 象限后接 5..8 象限），再截断到 num_classes。
    """
    if _FDI_LABEL_ORDER_OVERRIDE is not None:
        if len(_FDI_LABEL_ORDER_OVERRIDE) < num_classes:
            raise ValueError(
                f"FDI label order override has {len(_FDI_LABEL_ORDER_OVERRIDE)} entries, "
                f"but num_classes={num_classes}."
            )
        order = _FDI_LABEL_ORDER_OVERRIDE[:num_classes]
    else:
        order = default_fdi52_order()[:num_classes]
    return {f: i for i, f in enumerate(order)}
