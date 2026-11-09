"""Memory-only smoke: worst padded 1333x1333 batch, unchanged FP32 model."""
import torch

from b0.common import ARTIFACTS, build, json_write, load_checkpoint, model_args, seed_all
from util.misc import nested_tensor_from_tensor_list


def main():
    args=model_args();seed_all(args.seed)
    model,criterion,_,_=build(args)
    load_checkpoint(model,ARTIFACTS/f'checkpoints/smoke_b{args.batch_size}_a{args.accumulation_steps}/checkpoint.pth')
    model.train();criterion.train()
    # Opposite aspect ratios force the largest padding permitted by official
    # image preprocessing, without increasing any image's valid content size.
    images=[torch.randn(3,800,1333,device='cuda') if i%2==0 else
            torch.randn(3,1333,800,device='cuda') for i in range(args.batch_size)]
    targets=[{'labels':torch.tensor([0],device='cuda'),
              'boxes':torch.tensor([[.5,.5,.25,.25]],device='cuda')} for _ in images]
    samples=nested_tensor_from_tensor_list(images)
    torch.cuda.reset_peak_memory_stats()
    output=model(samples);losses=criterion(output,targets)
    loss=sum(value*criterion.weight_dict[key] for key,value in losses.items() if key in criterion.weight_dict)
    loss.backward();torch.cuda.synchronize()
    report={'passed':True,'scope':'synthetic_memory_only_no_detection_metrics',
        'batch_size':args.batch_size,'padded_shape':list(samples.tensors.shape),
        'loss_finite':bool(torch.isfinite(loss)),'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2}
    json_write(ARTIFACTS/'reports/memory_stress.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
