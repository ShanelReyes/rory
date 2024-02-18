from typing import Any, Optional, Union, Iterable

from .exceptions import OtherListElement


class Element(object):

    def __init__(self, value: Any, dllist: 'DLList'):
        """Element object is an element of the linked list. value
        keep track of the actual value that is set by the application.

        :param value: Actual value set by application.
        :param dllist: DLList of which this element is part of
        """
        #: Value in the element
        self.value: Any = value
        #: Link to previous element in the linked list
        self.prev: Optional[Element] = None
        #: Link to next element in the linked list
        self.next: Optional[Element] = None
        #: Reference to linked list the element is part of
        self.dllist: 'DLList' = dllist

    def nextelement(self) -> Optional['Element']:
        """Return the next list element or None"""
        p: Optional[Element] = self.next
        if p and p != p.dllist.root:
            return p

        return None

    def prevelement(self) -> Optional['Element']:
        """Return the next list element or None"""
        p: Optional[Element] = self.prev
        if p and p != p.dllist.root:
            return p

        return None


class DLList(object):
    """
    Object implementing a double linked list. Python verison of Golang list
    container. To simplify implementation, internally the list is implemented
    as a ring such that DLLlist.root is both next element of last and previous
    element of first
    """

    def __init__(self):
        #: Sentinel node of the list.
        self.root: Element = self._init()
        self.len: int = 0

    def _init(self) -> Element:
        """Creates an element for sentinel node"""
        e: Element = Element(None, self)
        e.next = e
        e.prev = e
        return e

    def __len__(self):
        return self.len

    def __iter__(self) -> Iterable[Element]:
        return self._iter(True)

    def _iter(self, value: bool) -> Iterable[Union[Element, Any]]:
        """Helper wrapper for both iter and iternodes"""
        front = self.front()
        while front:
            if value:
                yield front.value
            else:
                yield front
            front = front.nextelement()

    def _insert(self, e: Element, at: Element) -> Element:
        """Inserts e after at and increase len"""

        if e.dllist != self:
            raise OtherListElement(self, e)

        if at.dllist != self:
            raise OtherListElement(self, at)

        e.prev = at
        e.next = at.next
        e.prev.next = e
        if e.next:
            e.next.prev = e
        self.len = self.len + 1

        return e

    def _insertValue(self, value: Any, at: Element) -> Element:
        """Convenience wrapper for insert(Element(value), at)"""
        return self._insert(Element(value, self), at)

    def _remove(self, e: Element) -> Element:
        """Removes e from its list, decrements l.len, and return e"""
        if e.prev:
            e.prev.next = e.next
        if e.next:
            e.next.prev = e.prev
        e.next = None
        e.prev = None
        e.dllist = DLList()

        self.len = self.len - 1
        return e

    def _move(self, e: Element, at: Element) -> Element:
        """Moves e to next to at and return e"""
        if e == at:
            return e

        if e.prev:
            e.prev.next = e.next
        if e.next:
            e.next.prev = e.prev

        e.prev = at
        e.next = at.next

        if e.prev:
            e.prev.next = e
        if e.next:
            e.next.prev = e
        return e

    def reinit(self) -> None:
        """Reinits list"""
        self.root = self._init()
        self.len = 0

    def front(self) -> Optional[Element]:
        """return first element of list or None"""
        if not self.len:
            return None

        return self.root.next

    def back(self) -> Optional[Element]:
        """return last element of list or None"""
        if not self.len:
            return None

        return self.root.prev

    def appendleft(self, value: Any) -> Element:
        """Inserts a new element with value v at front of list. return Element


        :param value: Value to be saved
        :return: New element with value value
        """
        return self._insertValue(value, self.root)

    def append(self, value: Any) -> Element:
        """Inserts a new element with value v at tail of list. return Element

        :param value: Value to be saved
        :return: New element with value value
        """
        if self.root.prev:
            return self._insertValue(value, self.root.prev)
        raise RuntimeError("Should not reach here")

    def remove(self, e: Element) -> Any:
        """Remove element e and return value associated with it"""
        if e.dllist == self:
            self._remove(e)

        return e.value

    def insertbefore(self, value: Any, mark: Element) -> Element:
        """Inserts a new element e before mark and return e

        :param value: value to be saved
        :param mark: new element is added before mark
        """
        if mark.prev:
            return self._insertValue(value, mark.prev)
        raise RuntimeError("Should not reach here")

    def insertafter(self, value: Any, mark: Element) -> Element:
        """Inserts a new element e after mark and return e

        :param value: value to be saved
        :param mark: new element is added after mark
        """
        return self._insertValue(value, mark)

    def movetofront(self, e: Element) -> None:
        """Moves element e to front of list

        :param e: Element to move"""
        if e.dllist != self or self.root.next == e:
            return

        self._move(e, self.root)

    def movetoback(self, e: Element) -> None:
        """Moves element e to tail of list

        :param e: Element to move"""
        if e.dllist != self or self.root.prev == e:
            return

        if self.root.prev:
            self._move(e, self.root.prev)
            return

        raise RuntimeError("Should not reach here")

    def movebefore(self, e: Element, mark: Element) -> None:
        """Moves element e before mark

        :param e: Element to move
        :param mark: Element before which e will be moved"""
        if e.dllist != self or e == mark or mark.dllist != self:
            return

        if mark.prev:
            self._move(e, mark.prev)
            return

        raise RuntimeError("Should not reach here")

    def moveafter(self, e: Element, mark: Element) -> None:
        """Move element e after mark

        :param e: Element to move
        :param mark: Element after which e will be moved"""
        if e.dllist != self or e == mark or mark.dllist != self:
            return

        self._move(e, mark)

    def iternodes(self) -> Iterable[Element]:
        return self._iter(False)
