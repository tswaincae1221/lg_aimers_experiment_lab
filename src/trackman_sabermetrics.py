from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

GROUP_ORDER = [
    "release_repeatability",
    "velocity_retention",
    "spin_stability",
    "movement_repeatability",
    "repertoire_mix",
    "evidence_reliability",
]


GROUP_METADATA = {
    "release_repeatability": {
        "label_ko": "릴리스·익스텐션 재현성",
        "sabermetric_role": "제구 메커니즘의 직접 대리변수",
        "rationale": (
            "릴리스 높이·사이드·익스텐션의 과거 평균, 변동성, 전년 대비 변화로 "
            "투구 동작과 릴리스 포인트의 반복 가능성을 근사한다."
        ),
    },
    "velocity_retention": {
        "label_ko": "구속 유지·변동성",
        "sabermetric_role": "피로·메커니즘 변화 대리변수",
        "rationale": (
            "초속과 존 통과 속도의 수준, 표준편차, 전년 대비 변화로 구속 유지와 "
            "투구 동작 안정성을 근사한다."
        ),
    },
    "spin_stability": {
        "label_ko": "회전 안정성",
        "sabermetric_role": "공의 회전 품질·재현성 보조변수",
        "rationale": (
            "회전수의 과거 수준과 변동, 변화량을 사용한다. 회전수 자체는 제구가 "
            "아니므로 다른 재현성 지표와 함께 해석한다."
        ),
    },
    "movement_repeatability": {
        "label_ko": "무브먼트 재현성",
        "sabermetric_role": "공 움직임의 일관성 대리변수",
        "rationale": (
            "수직·수평 무브먼트의 과거 평균, 표준편차, 전년 대비 변화로 공 궤적의 "
            "반복 가능성과 변화 여부를 근사한다."
        ),
    },
    "repertoire_mix": {
        "label_ko": "구종 구성",
        "sabermetric_role": "구종 선택 맥락변수",
        "rationale": (
            "패스트볼·브레이킹·오프스피드·기타 구종 사용률이다. 직접적인 제구값은 "
            "아니지만 투수가 선택한 난이도와 접근법을 통제한다."
        ),
    },
    "evidence_reliability": {
        "label_ko": "표본·매핑 신뢰도",
        "sabermetric_role": "측정 신뢰도 보정변수",
        "rationale": (
            "매핑 신뢰도, 과거 투구 수, 관측 시차와 결측 여부로 Trackman 요약치의 "
            "신뢰도를 모델에 전달한다."
        ),
    },
}


def _measurement_role(feature: str) -> str:
    if feature.endswith("_std"):
        return "repeatability"
    if feature.startswith("tm_delta_"):
        return "year_over_year_change"
    if feature.endswith("_mean"):
        return "historical_level"
    if "usage_" in feature:
        return "pitch_mix"
    return "reliability"


def classify_trackman_feature(feature: str) -> str:
    """Map an engineered Trackman column to one non-overlapping sabermetric group."""
    if not feature.startswith("tm_"):
        raise ValueError(f"Not a Trackman feature: {feature}")
    if "usage_" in feature:
        return "repertoire_mix"
    if any(token in feature for token in ["rel_height", "rel_side", "extension"]):
        return "release_repeatability"
    if any(token in feature for token in ["rel_speed", "zone_speed"]):
        return "velocity_retention"
    if "spin_rate" in feature:
        return "spin_stability"
    if any(token in feature for token in ["induced_vert_break", "horz_break"]):
        return "movement_repeatability"
    return "evidence_reliability"


def build_trackman_catalog(features: Iterable[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        group = classify_trackman_feature(feature)
        metadata = GROUP_METADATA[group]
        rows.append(
            {
                "feature": feature,
                "group": group,
                "group_label_ko": metadata["label_ko"],
                "measurement_role": _measurement_role(feature),
                "sabermetric_role": metadata["sabermetric_role"],
                "rationale": metadata["rationale"],
            }
        )
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        return catalog
    order = {group: index for index, group in enumerate(GROUP_ORDER)}
    catalog["_group_order"] = catalog["group"].map(order)
    return catalog.sort_values(
        ["_group_order", "measurement_role", "feature"], ignore_index=True
    ).drop(columns="_group_order")


def group_trackman_features(catalog: pd.DataFrame) -> dict[str, list[str]]:
    if catalog.empty:
        return {}
    return {
        group: catalog.loc[catalog["group"] == group, "feature"].tolist()
        for group in GROUP_ORDER
        if (catalog["group"] == group).any()
    }
