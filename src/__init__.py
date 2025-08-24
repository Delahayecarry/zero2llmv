"""Zero2LLMV - 多模态大语言模型训练框架"""

__version__ = "0.2.0"
__author__ = "Zero2LLMV Team"
__description__ = "支持YAML配置的多模态大语言模型训练框架"

from .models import *
from .configs import *

__all__ = [
    "__version__",
    "__author__", 
    "__description__"
]