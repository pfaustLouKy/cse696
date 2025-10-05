from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService.save_account(
  token="ApiKey-ca4d785e-96de-4b79-870a-b7ae1b7fe230", # IBM Cloud API key.
  # Your token is confidential. Do not share your token in public code.
  instance="crn:v1:bluemix:public:quantum-computing:us-east:a/fe4c64141e620c6bbae892c86af03c75:d45f2eb9-68ce-466f-ab8d-601119e757ff::", # Optionally specify the instance to use.
  #plans_preference="['open', 'premium']", # Optionally set the types of plans to prioritize.  This is ignored if the instance is specified.
  # Additionally, instances of a certain plan type are excluded if the plan name is not specified.
  #region="us-east", # Optionally set the region to prioritize. Accepted values are 'us-east' or 'eu-de'. This is ignored if the instance is specified.
  #name="<account-name>", # Optionally name this set of account credentials. 
  #set_as_default=True, # Optionally set these as your default credentials.
)
