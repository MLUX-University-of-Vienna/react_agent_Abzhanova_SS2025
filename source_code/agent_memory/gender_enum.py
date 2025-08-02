from enum import Enum, auto

class GenderEnum(Enum):
    MALE      = auto()
    FEMALE    = auto()
    NEUTRAL   = auto()
    UNKNOWN   = auto()  

    def __str__(self):
        return {
            GenderEnum.MALE:    "male",
            GenderEnum.FEMALE:  "female",
            GenderEnum.NEUTRAL: "neutral",
            GenderEnum.UNKNOWN: "unknown",
        }[self]

