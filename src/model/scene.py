



class Scene:

    def __init__(self, gaussians, load_iteration=None):
        self.gaussians = gaussians
        self.load_iteration = load_iteration

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}