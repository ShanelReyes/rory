class ListException(Exception):
    pass


class OtherListElement(ListException):
    """Exception raised when you try to work with element of another list"""

    def __init__(self, ll, element):
        self.ll = ll
        self.element = element

    def __str__(self):
        return f"{self.element} not an element of {self.l}"
