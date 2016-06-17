
#TODO check axioms in unit test instead of in constructor


class FiniteGroup(object):

    def __init__(self, garray_type):

        if not isinstance(self, garray_type):
            raise TypeError('A subclass of FiniteGroup should derive from a subclass of GArray and pass'
                            ' that GArray subclass as garray_type to the FiniteGroup constructor.')
        self.garray_type = garray_type

        # Any subclass of FiniteGroup should also inherit from a subclass of GArray.
        # Assume the subclass has already called the constructor of the GArray subclass from which it is derived.
        if not self.shape[0] == self.size:
            raise ValueError('Group should be a flat GArray. Got shape ' + str(self.shape))

        # Check group axioms
        for g in self:
            # Inverse must be in G
            if not g.inv() in self:
                raise ValueError('FiniteGroup not closed under inverses: inv(' + str(g) + ') = ' + str(g.inv()))

            for h in self:
                if not g * h in self:
                    raise ValueError('FiniteGroup not closed under products: ' + str(g) + str(h) + ' = ' + str(g * h))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # Any instance of the same group should be equal
            return True
        else:
            return super(FiniteGroup, self).__eq__(other)

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            # Any instance of the same group should be equal
            return False
        else:
            return super(FiniteGroup, self).__ne__(other)
