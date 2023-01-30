from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from core import BertTC,PunctDataModule

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)    
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', '-bs', type=int, default=8)
    args = parser.parse_args()
    model = BertTC(args)

    best_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='best',
        save_on_train_epoch_end=False,
        save_top_k=1,
    )

    val_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='val-epoch={epoch:02d}-setp={step}-val_loss={val_loss:.5f}',
        save_last=True,
        save_on_train_epoch_end=False,
        save_top_k=3
    )

    early_stopping = EarlyStopping(
        'val_loss',
        mode='min',
        patience=4,
        check_on_train_epoch_end=False
    )

    trainer = Trainer.from_argparse_args(args, callbacks=[val_checkpoint_callback,early_stopping])
    datamodule = PunctDataModule(batch_size=args.batch_size)

    # trainer.test(model,ckpt_path='lightning_logs/version_7/checkpoints/val-epoch=epoch=01-setp=step=6103-val_loss=val_loss=0.39.ckpt', datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    
    try:
        trainer.test(ckpt_path='best', datamodule=datamodule)
    except:
        print("best ckpt not found, use the last ckpt")
        trainer.test('last', datamodule=datamodule)
