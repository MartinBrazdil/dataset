from time import time
from typing import List, AnyStr
from collections import defaultdict

from dataset.COCO.coco_parser import COCOParser
from dataset.utils.random_buffer import RandomBuffer
from torch.utils.data import DataLoader, Dataset, RandomSampler


class COCOImages(COCOParser, Dataset):
    class ImgBuf:
        def __init__(self, img_ann, img_pix):
            self.img_ann, self.img_pix = img_ann, img_pix

    def _new_torch_img_buf(self, swap_img_idx):
        # swap_img_id = self.img_ids[swap_img_idx]
        # img_ann = self.instances.loadImgs(swap_img_id)[0]
        img_ann = self.img_anns[swap_img_idx]
        img_bin = self.load_img_binary(img_ann)
        torch_img = self.load_torch_pixels(img_bin)
        return COCOImages.ImgBuf(img_ann, torch_img)

    def _new_PIL_img_buf(self, swap_img_idx):
        # swap_img_id = self.img_ids[swap_img_idx]
        # img_ann = self.instances.loadImgs(swap_img_id)[0]
        img_ann = self.img_anns[swap_img_idx]
        img_bin = self.load_img_binary(img_ann)
        pil_img = self.load_PIL_pixels(img_bin)
        return COCOImages.ImgBuf(img_ann, pil_img)

    def __init__(self, split: AnyStr, cats=List[AnyStr], buffer_size=200, swap_prob_offset=0., show_stats=False, *args, **kwargs):
        super().__init__({split}, *args, **kwargs)

        if split == 'TRAIN':
            self.instances = self.instances_train2017
            self.captions = self.captions_train2017
        elif split == 'TEST':
            self.instances = self.image_info_test2017
            self.captions = self.image_info_test_dev2017
        elif split == 'VAL':
            self.instances = self.instances_val2017
            self.captions = self.captions_val2017

        self.img_ids = []
        cat_ids = self.instances.getCatIds(cats)
        self.label_map, self.rev_label_map = {}, {}
        for cat_id in cat_ids:
            self.img_ids.extend(self.instances.catToImgs[cat_id])
        self.img_anns = self.instances.loadImgs(self.img_ids)

        self.buffer = RandomBuffer(buffer_size, swap_prob_offset, show_stats)
        self.buffer.fill((self._new_PIL_img_buf(idx) for idx in range(COCOImages.__len__(self))))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_buf = self.buffer[lambda: self._new_PIL_img_buf(idx)]
        while img_buf.img_pix.mode != 'RGB':
            return COCOImages.__getitem__(self, idx+1)
        return img_buf.img_ann, img_buf.img_pix

    @staticmethod
    def collate_fn(elems):
        anns_batch, pixs_batch = [], []
        for ann, pixs in elems:
            anns_batch.append(ann)
            pixs_batch.append(pixs)
        return {'anns_batch': anns_batch, 'pixs_batch': pixs_batch}


if __name__ == '__main__':
    dataset = COCOImages('VAL', ['person', 'car'], 100, 0.0, show_stats=True)
    sampler = RandomSampler(dataset, True, 10000)
    loader = DataLoader(dataset=dataset,
                        batch_size=8,
                        sampler=sampler,
                        num_workers=1,
                        collate_fn=dataset.collate_fn,
                        pin_memory=False)
    t0 = time()
    print(len(dataset))
    for b, batch in enumerate(loader):
        print('Batch[{}]: {}'.format(loader.batch_size, b))
    print('b={}, s={}, b/s={}'.format(b, (time() - t0), b / (time() - t0)))
