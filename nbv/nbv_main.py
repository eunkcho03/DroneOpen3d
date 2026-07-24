'''
List of things that has to be done
1. Compute candidate positions
2. Compute gain for each position
    2.1 Compute camera view from candidate position
    2.2 Transform surface coordinates to camera coordinte
    2.3 Counte the number of unknown voxels to be revealed. 
3. Store each gain value per candidate position
'''


class NBVPlanner: 
    def __init__(self, fov, image_size, num_canidates):
        self.fov = fov
        self.image_size = image_size
        self.num_candidates = 