# General functionality needed across multiple modules
# Note that this is a good place to place code that needs to be imported at
# both the top level and within a solver class so that it can be imported
# without circular dependencies

def get_transform_args(val, axis):
    """Converts an axis and value into a yaw, pitch, roll, and dispacement"""
    if axis in ["x", "y", "z"]:
        yaw, pitch, roll = 0., 0., 0.
        if axis == "x":
            displacement = [val, 0., 0.]
        elif axis == "y":
            displacement = [0., val, 0.]
        elif axis == "z":
            displacement = [0., 0., val]
    else:
        displacement = [0., 0., 0.]
        if axis == "yaw":
            yaw, pitch, roll = val, 0., 0.
        elif axis == "pitch":
            yaw, pitch, roll = 0., val, 0.
        elif axis == "roll":
            yaw, pitch, roll = 0., 0., val
    return yaw, pitch, roll, displacement


def get_id(taken_set, max_val, min_val=1):
    # Creates an integer ID from within the set without duplicates and while
    # filling the whole range

    # This method will prefer the cheapest option (take next available)
    # over the more expensive filling in the holes
    id_ = min_val
    max_in_set = max(taken_set)
    if max_in_set + 1 > max_val:
        # Then we need to go and fill holes
        for i in range(min_val, max_val + 1):
            if i not in taken_set:
                id_ = i
                break
        else:
            # We didnt find the ID. Raise an Error
            raise ValueError("No IDs Available <= {}".format(max_val))
    else:
        # Just take the next value then
        id_ = max_in_set + 1

    # Update the set and return the id
    taken_set.add(id_)

    return id_
