from dataclasses import dataclass

@dataclass
class User:
    email: str
    password: str
    isAdmin: bool

@dataclass
class fraud:
    time: int
    amount: float

