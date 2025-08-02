from enum import Enum, auto

class Gender(Enum):
    MALE      = auto()
    FEMALE    = auto()
    NEUTRAL   = auto()
    UNKNOWN   = auto()  

    def __str__(self):
        return {
            Gender.MALE:    "male",
            Gender.FEMALE:  "female",
            Gender.NEUTRAL: "neutral",
            Gender.UNKNOWN: "unknown",
        }[self]

