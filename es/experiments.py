
"""
Details of all the experiments we run. 
We do not seek to tune these parameters too much.
The parameters here work for baselines.
"""

def get_experiment(params):
	if params['env_name'] in ['HalfCheetah-v2','HalfCheetah-v1']:
		params['layers'] = 0
		params['sensings'] = 200
		params['learning_rate'] = 0.005
		params['shift'] =0
		params['sigma'] = 0.1
		params['steps'] = 1000
	elif params['env_name'] in ['Walker2d-v2']:
		params['layers'] = 0
		params['sensings'] = 100
		params['learning_rate'] = 0.05
		params['shift'] =0
		params['sigma'] = 0.1
		params['steps'] = 1000
	elif params['env_name'] == 'Swimmer-v2':
		params['layers'] = 0
		params['sensings'] = 50
		params['learning_rate'] = 0.05
		params['shift'] = 0
		params['sigma'] = 0.1
		params['steps'] = 1000
	elif params['env_name'] == 'BipedalWalker-v2':
		params['layers'] = 0
		params['sensings'] = 100
		params['learning_rate'] = 0.01
		params['shift'] = 0
		params['sigma'] = 0.1
		params['steps'] = 1000
	return(params)

