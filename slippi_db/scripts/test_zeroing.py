import numpy as np
from typing import NamedTuple
from slippi_ai import types

def zero_out_namedtuple(nt: NamedTuple) -> NamedTuple:
    def zero_out_field(field):
        if isinstance(field, np.ndarray):
            return np.zeros_like(field)  # Maintain shape and type
        elif isinstance(field, (np.uint16, np.bool_, np.float32, np.uint8)):
            return type(field)(0)  # Return zero of the same type
        elif isinstance(field, tuple) and hasattr(field, "_fields"):  # Check for NamedTuple
            return zero_out_namedtuple(field)  # Recursively zero out subfields
        else:
            return field  # Return as is for non-handled types

    zeroed_fields = {key: zero_out_field(getattr(nt, key)) for key in nt._fields}
    return type(nt)(**zeroed_fields)

# Example usage
buttons = types.Buttons(
    A=np.bool_(True),
    B=np.bool_(False),
    X=np.bool_(True),
    Y=np.bool_(False),
    Z=np.bool_(True),
    L=np.bool_(False),
    R=np.bool_(True),
    D_UP=np.bool_(False),
)

stick_main = types.Stick(x=np.float32(0.75), y=np.float32(-0.33))
stick_c = types.Stick(x=np.float32(-0.5), y=np.float32(0.4))

controller = types.Controller(
    main_stick=stick_main,
    c_stick=stick_c,
    shoulder=np.float32(0.8),
    buttons=buttons,
)

player = types.Player(
    percent=np.uint16(150),
    facing=np.bool_(True),
    x=np.float32(1.25),
    y=np.float32(-1.75),
    action=np.uint16(5),
    invulnerable=np.bool_(False),
    character=np.uint8(2),
    jumps_left=np.uint8(3),
    shield_strength=np.float32(40.0),
    on_ground=np.bool_(True),
    is_dead=np.bool_(False),
    controller=controller,
)

zeroed_player = zero_out_namedtuple(player)
print(player)
print(zeroed_player)