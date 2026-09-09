Currently, the attribution graph can only show the attribution of features and tokens to a target logit. Recent work suggest we attribute to a concept instead, https://arxiv.org/abs/2608.27510 , and use that to infer the internal reasoning of the model.

My current approach to summarize the attribution graph is to measure the features relevance and discard them. But I don't think there's a universal way to measure relevance. The relevance of features should be conditioned on the mechanistic claim that we are making.

For example if we ask the model

You're a doctor, your job is to diagnose the patient.
patient: I have a fever, I have a headache, I'm pregnant ...

The attribution graph can be really big, and we can ask multiple questions:
- How does the doctor role effect the model response
- Does the model have an idea of what the patient might be suffering from

Does attributing to a concept alone eliminate the need to measure relevance?