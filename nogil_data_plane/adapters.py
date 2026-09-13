"""Explicit adapters for alpha-beta-CROWN domain packet types."""
from __future__ import annotations
from .domain_packet_codec import DomainPacketCodec, dataclass_adapter

def register_alpha_beta_crown_adapters(codec: DomainPacketCodec) -> DomainPacketCodec:
    """Register fixed local constructors; never import a peer-provided module."""
    from heuristics.decision_types import BatchFirstBranchingDecisions
    from state.alpha import AlphaFullInfoData, AlphaValueData
    from state.beta import BetaFullData, BetaValues, NumsEffectiveBetasPerDomain
    from state.lA import BatchedlA
    from domain_clipper import ClipDecisions, SubDomainClipDecisions
    classes = (
        (BatchFirstBranchingDecisions, "abcrown.batch_first_branching_decisions"),
        (AlphaValueData, "abcrown.alpha_value_data"),
        (AlphaFullInfoData, "abcrown.alpha_full_info_data"),
        (BetaFullData, "abcrown.beta_full_data"),
        (NumsEffectiveBetasPerDomain, "abcrown.nums_effective_betas"),
        (BetaValues, "abcrown.beta_values"),
        (BatchedlA, "abcrown.batched_la"),
        (ClipDecisions, "abcrown.clip_decisions"),
        (SubDomainClipDecisions, "abcrown.sub_domain_clip_decisions"),
    )
    for python_type, tag in classes:
        codec.register_adapter(dataclass_adapter(python_type, tag))
    return codec
