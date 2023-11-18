from torch.utils.tensorboard import SummaryWriter
import torch
import time


def record_process(opt, shared_data):
	recorder = Recorder(opt, shared_data)
	recorder.run()

class Recorder:
	'''Because the running curve written by evaluator can be unsorted,
	we use a Recorder process to sort the running curve point and record it with tensorboard'''
	def __init__(self, opt, shared_data):
		self.shared_data = shared_data
		self.writer = SummaryWriter(log_dir=opt.writepath)

	def run(self):
		# recorder will be terminated in main process
		while True:
			time.sleep(60)
			curve = self.shared_data.get_curve()

			if len(curve) == 0: pass
			else:
				curve = torch.tensor(curve)
				score, steps, walltime = curve[:, 0], curve[:, 1], curve[:, 2]

				# sort
				steps, sort_ind = torch.sort(steps)
				score = score[sort_ind]
				walltime = walltime[sort_ind]

				for _ in range(len(curve)):
					self.writer.add_scalar('ep_r', score[_], steps[_], walltime[_])




