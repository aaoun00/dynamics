import numpy as np

def get_index(name, FIRA):
    """
    Returns the index of a given event name in the FIRA structure.
    """
    for item in FIRA[0, 0][0, 0]:
        if len(item.shape) > 1 and item.shape[1] == 65:
            for j in range(item.shape[1]):
                if item[0][j] == name:
                    return j
    raise ValueError('Name not found')

def format_event(event):
    """
    Formats an event to a float
    """
    for i in range(len(event)):
        if event[i].shape[1] == 0:
            event[i] = float('nan')
        else:
            event[i] = float(event[i][0][0])
    return np.array(event, dtype=float)

