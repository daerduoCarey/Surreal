# ***Surreal*** initiative

## Goal

- Establish a shared codebase of state-of-the-art algorithm implementations and reusable utilities among Stanford RL group members.

- Accelerate research by ...

	- Providing strong baseline implementation for rapid idea iteration.
	- Reducing potential bug dimensions.
	- Measuring performance in a reproducible fashion.
	- Driving continual improvement of benchmarks.
	- Maintaining readable code for all future publications. 


## Maintenance

- Unit testing for the reusable parts and the shared utility scripts.
- End-to-end training/testing for algorithm benchmarks. 
- Separate `master` branch and `stable` branch. The `master` branch is allowed to be more exploratory, but the `stable` branch should pass all tests (unit + end2end) at all time. `master` will be merged into `stable` on a regular basis. 
- Unless it’s a trivial documentation change, one must submit a pull request instead of pushing directly to the `master`/`stable` branches. The pull request should have a detailed description of the changes, ideally as bullet points. It must receive at least one "Looks Good To Me (LGTM)" comment before it can be merged. 
- We can create a special folder on the master branch called `sandbox` for people to share their latest code features, which might not have been fully tested. One should push to a subfolder, e.g. `sandbox/<username>/`. 
- We also need a systematic way to keep track of the best set of hyperparameters for each benchmark.


## Getting started

### Dependencies

- Python 3
	- It's time to move on. We will only support python 3 and please feel free to use the backward-incompatible language features.
	- Recommended python package manager: [Anaconda](https://www.continuum.io/downloads). 
	- Anaconda has its own virtualenv wrapper: [setup tutorial](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/). 

- Tensorflow
	- [Installation](https://www.tensorflow.org/get_started/os_setup) 
	- [API reference](https://www.tensorflow.org/api_docs/python/)
	- [Model zoo](https://github.com/tensorflow/models)

- [OpenAI Gym](https://github.com/openai/gym): standardized [API](https://gym.openai.com/docs) for agents to interact with a diverse range of RL environments. 

- [OpenAI Universe](https://universe.openai.com/)
	- A VNC-powered platform for evaluating and training intelligent agents across a huge variety of games, websites and other applications.
	- The [blog release](https://openai.com/blog/universe/) provides more context. 

- [Keras](https://keras.io/): convenient Tensorflow wrapper to build network architectures quickly. 

### Code structure

[ToDo]

### Discussion

Slack channel: [StanfordRL](stanfordrl.slack.com)
