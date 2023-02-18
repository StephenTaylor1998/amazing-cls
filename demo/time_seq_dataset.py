# import gif
# from matplotlib import pyplot as plt
import gif
import numpy as np
from itertools import permutations, combinations

# perm = permutations(range(4))
from matplotlib import pyplot as plt

comb = combinations(range(16), 12)

# print(list(perm))
print(len(list(comb)))


class FIG:
    def __init__(self, source):
        self.source = source
        self.frames = [self.generate_frame(image) for image in source]

    def save(self, path, duration=100, unit="milliseconds",
             between="frames", loop=True):
        gif.save(self.frames, path, duration, unit, between, loop)

    @gif.frame
    def generate_frame(self, image):
        plt.imshow(image)


if __name__ == '__main__':
    bg = np.zeros((16, 16, 16))
    # img = bg + np.expand_dims(np.eye(16), axis=2)
    # FIG(img).save('./example1.gif')
    # img = bg + np.expand_dims(np.eye(16), axis=1)
    # FIG(img).save('./example2.gif')
    # img = bg + np.expand_dims(np.eye(16), axis=2)[::-1]
    # FIG(img).save('./example3.gif')
    # img = bg + np.expand_dims(np.eye(16), axis=1)[::-1]
    # FIG(img).save('./example4.gif')

    img = bg + np.expand_dims(np.eye(16), axis=2) @ np.expand_dims(np.eye(16), axis=1)
    FIG(img).save('./example5.gif')
    img = bg + np.expand_dims(np.eye(16), axis=2) @ np.expand_dims(np.eye(16), axis=1)[::-1]
    FIG(img).save('./example6.gif')
    img = bg + np.expand_dims(np.eye(16), axis=2)[::-1] @ np.expand_dims(np.eye(16), axis=1)
    FIG(img).save('./example7.gif')
    img = bg + np.expand_dims(np.eye(16), axis=2)[::-1] @ np.expand_dims(np.eye(16), axis=1)[::-1]
    FIG(img).save('./example8.gif')




