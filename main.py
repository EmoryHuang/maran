import utils
from args import get_args
from dataloader import Poidataloader
from model import *
from trainer import Trainer

# load config file
args = get_args()

# init logger
logger = utils.init_logger()

# init seed
utils.init_seed(3407)


def main():
    logger.info('start loading checkin data...')
    poi_loader = Poidataloader(args)
    # Gowalla / Foursquare
    if not args.dataset_test_path.exists():
        checkins = poi_loader.load(args.dataset, args.dataset_file)
    logger.info('loading checkin data done!')

    logger.info('start creating dataset...')
    poi_loader.create_dataset(args.mode, args.dataset)
    config = poi_loader.config
    logger.info('creating dataset done!')

    logger.info('start loading model...')
    model = PoiModel(config)
    logger.info('loading model done!')

    trainer = Trainer(config=config, logger=logger, gpu=config.gpu)
    if config.mode == 'train':
        trainer.train(model=model, dataloader=poi_loader)
    else:
        trainer.test(model, dataloader=poi_loader, model_path=config.model_path)


if __name__ == '__main__':
    main()

# train
# python -u main.py --mode=train --dataset=Gowalla --gpu=1
# nohup python -u main.py --mode=train --dataset=Gowalla --gpu=1 > ./go.log 2>&1 &
# nohup python -u main.py --mode=train --dataset=Yelp --gpu=2 --mask=False > ./yelp.log 2>&1 &

# test
# python main.py --mode=test --dataset=Gowalla --model_path='./Model/model_gowalla.pkl' --gpu=1
