from typing import cast
import unittest

from lfu_cache import DLList, Element, OtherListElement

class TestDLList(unittest.TestCase):

    def checkpointers(self, l, es):
        root = l.root

        self.assertEqual(len(l), len(es), msg="Passed elements count dont match with list count")

        # zero length lists must be the zero value or properly initialized (sentinel circle)
        if not len(es):
            self.assertTrue(l.root.next is not None and l.root.next == root)
            self.assertTrue(l.root.prev is not None and l.root.prev == root)

        for i, e in enumerate(es):
            prev = root
            Prev = None

            if i > 0:
                prev = es[i-1]
                Prev = prev

            p = e.prev
            self.assertEqual(p, prev)

            self.assertEqual(e.prevelement(), Prev)

            next = root
            Next = None
            if i < len(es) - 1:
                next = es[i+1]
                Next = next

            n = e.next
            self.assertEqual(n, next)

            self.assertEqual(e.nextelement(), Next)

    def test_dllist(self):
        l = DLList()
        # Empty list
        self.checkpointers(l, [])

        # Single elmenent list
        e = l.appendleft("a")
        self.checkpointers(l, [e])
        l.movetofront(e)
        self.checkpointers(l, [e])
        l.movetoback(e)
        self.checkpointers(l, [e])
        l.remove(e)
        self.checkpointers(l, [])


        # Multiple element
        e2 = l.appendleft(2)
        e1 = l.appendleft(1)
        e3 = l.append(3)
        e4 = l.append("banana")

        self.assertEqual(l.back(), e4)

        self.checkpointers(l, [e1, e2, e3, e4])

        self.assertEqual(l.back(), e4)

        l.remove(e2)
        self.checkpointers(l, [e1, e3, e4])

        # move from between
        l.movetofront(e3)
        self.checkpointers(l, [e3, e1, e4])

        l.movetofront(e1)
        self.checkpointers(l, [e1, e3, e4])

        # move from middle
        l.movetoback(e3)
        self.checkpointers(l, [e1, e4, e3])

        # move from back
        l.movetofront(e3)
        self.checkpointers(l, [e3, e1, e4])

        # no-op
        l.movetofront(e3)
        self.checkpointers(l, [e3, e1, e4])

        # move from front
        l.movetoback(e3)
        self.checkpointers(l, [e1, e4, e3])

        # should be no op
        l.movetoback(e3)
        self.checkpointers(l, [e1, e4, e3])


        # insert before front
        e2 = l.insertbefore(2, e1)
        self.checkpointers(l, [e2, e1, e4, e3])

        l.remove(e2)

        # insert before middle
        e2 = l.insertbefore(2, e4)
        self.checkpointers(l, [e1, e2, e4, e3])

        l.remove(e2)

        # insert before back
        e2 = l.insertbefore(2, e3)
        self.checkpointers(l, [e1, e4, e2, e3])

        l.remove(e2)
        self.checkpointers(l, [e1, e4, e3])

        # test iteration
        sum = 0
        for e in l:
            try:
                sum = sum + e
            except TypeError as e:
                pass
        self.assertEqual(sum, 4, msg="Total of int in list should be 4")

    def test_remove(self):
        l = DLList()

        e1 = l.append(1)
        e2 = l.append(2)

        self.checkpointers(l, [e1, e2])

        e = l.front()
        l.remove(e)
        self.checkpointers(l, [e2])

        # removing removed element
        l.remove(e)
        self.checkpointers(l, [e2])


    def test_removeincorrectlistitems(self):
        l1 = DLList()
        l1.append(1)
        l1.append(2)

        l2 = DLList()
        l2.append(1)
        l2.append(2)

        e = l1.front()
        l2.remove(e)

        self.assertEqual(len(l2), 2)


    def test_removedelementnextprev(self):
        l = DLList()
        l.append(1)
        l.append(2)

        e = l.front()
        l.remove(e)

        self.assertEqual(len(l), 1)
        self.assertEqual(e.nextelement(), None)
        self.assertEqual(e.prevelement(), None)


    def test_move(self):
        l = DLList()

        e1 = l.append(1)
        e2 = l.append(2)
        e3 = l.append(3)
        e4 = l.append(4)

        l.moveafter(e3,e3)
        self.checkpointers(l, [e1, e2, e3, e4])

        l.movebefore(e2, e2)
        self.checkpointers(l, [e1, e2, e3, e4])

        l.moveafter(e3, e2)
        self.checkpointers(l, [e1, e2, e3, e4])

        l.movebefore(e2, e3)
        self.checkpointers(l, [e1, e2, e3, e4])

        l.movebefore(e2, e4)
        self.checkpointers(l, [e1, e3, e2, e4])
        e2, e3 = e3, e2

        l.movebefore(e4, e1)
        self.checkpointers(l, [e4, e1, e2, e3])

        e1, e2, e3, e4 = e4, e1, e2, e3
        l.moveafter(e4, e1)
        self.checkpointers(l, [e1, e4, e2, e3])

        e2, e3, e4 = e4, e2, e3
        l.moveafter(e2, e3)
        self.checkpointers(l, [e1, e3, e2, e4])


    def test_unknownmark(self):
        l = DLList()

        e1 = l.append(1)
        e2 = l.append(2)
        e3 = l.append(3)

        self.assertRaises(OtherListElement, l.insertafter, 1, Element(None, DLList()))
        self.assertRaises(RuntimeError, l.insertbefore, 1, Element(None, DLList()))
