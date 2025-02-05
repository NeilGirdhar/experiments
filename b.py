dclass Point:
    x: int
    y: int = field(default=1, converter=int)

    # Exists by default, but can be overridden.
    # All constructors are class-methods that return Self.
    # init is special and is called by Point(x, y) or Point.init(x, y)
    @constructor
    def init(cls, x: int, y: int = 1):
        return cls.__new__(x, y)  # __new__ exists on every dclass and just accepts the parameters.

    # Called by Point.on_diagonal(z)
    @constructor
    def on_diagonal(cls, z: int, /):
        return cls.__new__(z, z)

    # Post-init is called after every constructor.
    def __post_init__(self):
        assert self.y > self.x


dclass 3DPoint(Point):
    # x: float (disallowed by LSP)
    # x: Literal[0, 1] (disallowed unless x is ReadOnly)
    z: int

    # InitVar is disallowed since the init constuctor can be specified instead with whatever
    # parameters you want.  It would also complicate inheritance too much since you would be
    # inclined to forward these to super.  In such case, you should instead take a point instance in
    # the constructor and use that to initialize yourself.

    # init=False on the field is likewise disallowed since you can simply not specify it in the
    # constructor.

    @constructor
    def init(cls, x: int, y: int = 1, *, z: int):
        return cls.__new__(x, y, z)

    # Constructors are never inherited.
    @constructor
    def on_diagonal(cls, z: int, /):
        return cls.__new__(z, z, z)

    # Constructors are never inherited.
    @constructor
    def on_diagonal_alternative(cls, z: int, /):
        point = Point.on_diagonal(z)
        return cls.__new__(as_dict(point), z)

    # Implicitly calls super().__post_init__()
    def __post_init__(self):
        assert self.z > self.x


# Type checker should raise on calls to __init__ and __new__, which should be considered an internal
# detail:
# 3DPoint.__init__(...)
# 3DPoint.__new__(...)


# Advantages:
# * No weird issues with inheritance:
#   * You can never forget to call super in a constructor since a constructor must initialize every
#     member, so it doesn't need to call super.
#   * You can never forget to call super in __post_init__ since it always Implicitly calls super.
#   * No issues with constructors disobeying LSP since constructors are simply not inherited.
