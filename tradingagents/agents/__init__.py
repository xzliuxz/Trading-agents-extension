from __future__ import annotations
__all__ = ['create_macro_analyst']

def __getattr__(name: str):
    if name == 'create_macro_analyst':
        from .macro.macro_agent import create_macro_analyst
        return create_macro_analyst
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
