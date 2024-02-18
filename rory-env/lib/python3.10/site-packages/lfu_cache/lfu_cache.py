from typing import Dict, Any, Optional, cast, Union
from .dllist import DLList, Element


class _NodeItem(object):
    """Object kept in frequency list node list"""

    def __init__(self, key: Any, value: Any, freqitem: '_FrequencyItem'):
        #: key set by by user for cache item
        self.key: Any = key
        #: value for key set by user for cache item
        self.value: Any = value
        #: FrequencyItem to which this node belongs
        self.freqitem: '_FrequencyItem' = freqitem


class _FrequencyItem(object):
    """Object kept as value inside LFUCache's frequency list. This is
    the type that is stored in each DLList Element"""

    def __init__(self, frequency: int):
        #: Frequency of this item
        self.frequency: int = frequency
        #: Linked list of nodelist. Items for this list will be NodeItem
        self.nodelist: DLList = DLList()
        #: Parent element to which this item belongs
        #: we use this to get the next frequency item
        self.element: Optional[Element] = None


class LFUCache(object):
    """Object implents an O(1) algorithm for implementing the LFU
    cache eviction scheme. This is based on paper at:
    http://dhruvbird.com/lfu.pdf

    See: https://en.wikipedia.org/wiki/Least_frequently_used"""

    def __init__(self, limit: int):
        #: Limit of cache before an element is evicted
        self.limit: int = limit
        #: We use a dict to store the data. Value is element from the
        #: corresponding frequency list item
        self.cache: Dict[Any, Element] = {}
        #: Frequency list
        self.freqlist: DLList = DLList()

        if self.limit < 0:
            self.limit = 0

    def _movetonextfrequency(self, element: Element) -> Element:
        """Element is the linked list wrapper of Node item
           element.value is the actual NodeItem whose value
           has been updated to latest. Now we need it to move to
           next frequency item."""

        nodeitem = cast(_NodeItem, element.value)
        assert nodeitem.freqitem.nodelist == element.dllist

        # Get the next element in the frequency list
        # nodeitem has reference to the frequency item it belongs to
        # from there we get its list element and get the next frequency
        currentfrequency = nodeitem.freqitem.frequency
        assert nodeitem.freqitem.element
        nodefreqitemelement = nodeitem.freqitem.element

        nextfrequencyitem = nodefreqitemelement.nextelement()
        # next item has to be present and its frequency is more then current
        # to be used. otherwise we just add an item after that.
        # Cases:
        # 1 -> 3
        # 4 -> None
        # in the first case 3 is present after 1 but we need it to be 2
        if (
            nextfrequencyitem and
            cast(
                _FrequencyItem,
                nextfrequencyitem.value).frequency == currentfrequency + 1
        ):
            nextfrequency = cast(_FrequencyItem, nextfrequencyitem.value)
        else:
            nextfrequency = _FrequencyItem(currentfrequency + 1)
            newfrequencyitemelement = self.freqlist.insertafter(
                nextfrequency, nodefreqitemelement)
            nextfrequency.element = newfrequencyitemelement

        nodeitem.freqitem.nodelist.remove(element)

        # we also remove this freqitem if its empty
        if not len(nodeitem.freqitem.nodelist):
            self.freqlist.remove(nodeitem.freqitem.element)

        newelement = nextfrequency.nodelist.append(
            _NodeItem(nodeitem.key, nodeitem.value, nextfrequency))

        return newelement

    def _evictlfu(self):
        """Removes the first element from frequency list and cache"""
        first = self.freqlist.front()
        if first:
            frequencyitem = cast(_FrequencyItem, first.value)
            element = frequencyitem.nodelist.front()
            nodeitem = cast(_NodeItem, element.value)
            frequencyitem.nodelist.remove(element)

            del self.cache[nodeitem.key]

    def put(self, key: Any, value: Any):
        """
        Puts an object to the cache
        """
        # We dont even attempt zero limit cache
        if not self.limit:
            return

        nodeitemelement = self.cache.get(key)
        firstfreq = self.freqlist.front()

        # Evict if key was not found and we have reached element
        # otherwise we will remove existing one which we dont want even if we
        # have reached the length
        if nodeitemelement is None and len(self.cache) == self.limit:
            self._evictlfu()

        if nodeitemelement:
            # we update the value before it can be moved to next element
            cast(_NodeItem, nodeitemelement.value).value = value
            newelement = self._movetonextfrequency(nodeitemelement)
        elif firstfreq and cast(_FrequencyItem, firstfreq.value).frequency == 0:
            # If the first node is 0
            f = cast(_FrequencyItem, firstfreq.value)
            newnodeitem = _NodeItem(key, value, f)
            newelement = f.nodelist.append(newnodeitem)
        else:
            # We need a freqitem with 0 and add the element to it
            zerofreqitem = _FrequencyItem(0)
            freqitemelement = self.freqlist.appendleft(zerofreqitem)
            zerofreqitem.element = freqitemelement
            newnodeitem = _NodeItem(key, value, zerofreqitem)
            newelement = zerofreqitem.nodelist.append(newnodeitem)

        self.cache[key] = newelement

    def get(self, key) -> Union[KeyError, Any]:
        """Gets value by key. Raises KeyError, if not found"""
        element = self.cache[key]  # this will just return KeyError if not ther
        nodeitem = cast(_NodeItem, element.value)
        value = nodeitem.value

        newelement = self._movetonextfrequency(element)
        self.cache[key] = newelement

        return value
