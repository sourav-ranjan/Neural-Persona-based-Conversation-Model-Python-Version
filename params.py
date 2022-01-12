class params:
	def __init__(self):
		self.batch_size=256
		self.dimension=512
		self.dropout=0.2
		self.max_iter=10
		self.layers=4
		self.SpeakerNum=866
		self.PersonaMode=True
		
		self.train_path="data/testing"
		self.train_file="/dataset_full_output_int_train.txt"
		self.dev_file="/dataset_full_output_int_valid.txt"
		self.dictPath="/vocabulary"
		self.speakdictPath="/dataset_full_characters.txt"
		
		self.fine_tuning=False
		self.fine_tuning_model="save/model"
		if self.PersonaMode:
			self.saveFolder="save/"+self.train_path.split("/")[-1]     # save/testing
		else:
			self.saveFolder="save/"+self.train_path.split("/")[-1]+"/non_persona"    # save/testing/non_persona
			
		self.init_weight=0.1
		self.alpha=1
		self.start_halve=6
		self.max_length=100
		self.vocab_source=25010
		self.vocab_target=25010
		self.vocab_dummy=25006
		self.thres=5
		self.source_max_length=50
		self.target_max_length=50
		
		self.save_prefix=self.saveFolder+"/model"
		self.save_params_file=self.saveFolder+"/params"
		self.output_file=self.saveFolder+"/log"    # save/testing/log or save/testing/non_persona/log
		
		self.reverse=False
		self.reverse_target=False
		self.saveModel=True
		self.use_GPU=True
