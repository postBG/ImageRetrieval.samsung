import torch
from tqdm.notebook import tqdm


class Database(object):
    def __init__(self, dataloader, feature_extractor):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.feature_extractor = feature_extractor
        self.images, self.extracted_features, self.targets = self.extract_features_to_construct_database()

    @torch.no_grad()
    def extract_features_to_construct_database(self):
        self.feature_extractor.eval()
        all_images = []
        all_features = []
        all_targets = []
        for xs, ys in tqdm(self.dataloader):
            features = self.feature_extractor(xs)
            features = features.view(features.size(0), -1)
            all_features.extend(features)

            xs = xs.view(xs.size(0), -1)
            all_images.extend(xs)

            all_targets.extend(ys)
        return torch.stack(all_images), torch.stack(all_features), torch.stack(all_targets)

    def get_image_and_target(self, idx):
        return self.dataset[idx]
