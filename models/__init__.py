"""Model zoo for the CAPTCHA-X journal upgrade.

Import concrete classes from their modules, e.g. `models.crnn_clean`, to keep
module-level smoke runs free of eager-import side effects.
"""

__all__ = [
    "CaptchaCRNN",
    "CRNNClean",
    "ConvNeXtCaptchaCTC",
    "FixedSlotCaptchaCNN",
    "ViTCaptchaCTC",
]


def __getattr__(name: str):
    if name in {"CaptchaCRNN", "CRNNClean"}:
        from .crnn_clean import CaptchaCRNN, CRNNClean

        return {"CaptchaCRNN": CaptchaCRNN, "CRNNClean": CRNNClean}[name]
    if name == "ConvNeXtCaptchaCTC":
        from .convnext_captcha import ConvNeXtCaptchaCTC

        return ConvNeXtCaptchaCTC
    if name == "FixedSlotCaptchaCNN":
        from .fixed_slot import FixedSlotCaptchaCNN

        return FixedSlotCaptchaCNN
    if name == "ViTCaptchaCTC":
        from .vit_captcha import ViTCaptchaCTC

        return ViTCaptchaCTC
    raise AttributeError(name)
