import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np
from visualiser import Visualiser
import torch
import tqdm


def cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0






from visdom import Visdom

class Visualizations:
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
            
            

from datetime import datetime
import visdom

class Visualizations2:
    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win='loss',
            update='append' if 'loss' else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 10 steps)',
            )
        )

        


    











def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None,
          num_classes=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)
    
    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    
    vis = Visualizations()
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        
        
        
        tl = train_loader
        
        try:
            mean_loss = 0
          
            for i, (inputs, targets) in enumerate(tl):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = cuda(targets)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                

                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                # Visualization data
                #vis.plot_loss(np.mean(mean_loss), step)
                
                # valid_losses.clear()
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)  
            #vis.plot_loss(np.mean(valid_losses), step)
            
      
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
        
        vis.plot('loss', 'train', 'Loss per epoch '+args.model, epoch, mean_loss)
        vis.plot('loss', 'validation', 'Loss per epoch '+ args.model, epoch, valid_loss)
        


    
        