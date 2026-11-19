"""Utilities for the AWS code environment."""

import time
from google.cloud import translate_v2 as translate

from src.config import get_secrets, google_service_account_info  # noqa: F401  (re-exported)



def list_experiment_runs(sm, experiment_name):
    """Print names of all trial runs associated with a SageMaker experiment.

    Parameters
    ----------
    sm : boto3 SageMaker session
        A boto3 SageMaker session created with boto3.Session().client('sagemaker').
    experiment_name : str
        The experiment name to print trial runs from.
    """
    trials = sm.list_trials(ExperimentName=experiment_name)["TrialSummaries"]
    print("TrialNames:")
    for trial in trials:
        trial_name = trial["TrialName"]
        print(f"\n{trial_name}")

        components_in_trial = sm.list_trial_components(TrialName=trial_name)
        print("\tTrialComponentNames:")
        for component in components_in_trial["TrialComponentSummaries"]:
            component_name = component["TrialComponentName"]
            print(f"\t{component_name}")


def delete_trial_run(sm, experiment_name, run_name):
    """Delete a specific trial run associated with a SageMaker experiment.

    Parameters
    ----------
    sm : boto3 SageMaker session
        A boto3 SageMaker session created with boto3.Session().client('sagemaker').
    experiment_name : str
        The experiment name to remove trial runs from.
    run_name : str
        The trial run to be removed from the experiment.
    """
    run_found = False
    trials = sm.list_trials(ExperimentName=experiment_name)["TrialSummaries"]
    print("TrialNames:")
    for trial in trials:
        trial_name = trial["TrialName"]
        # print(f"\n{trial_name}")

        components_in_trial = sm.list_trial_components(TrialName=trial_name)
        # print('\tTrialComponentNames:')
        for component in components_in_trial["TrialComponentSummaries"]:
            component_name = component["TrialComponentName"]
            # print(f"\t{component_name}")
            if component_name == run_name:
                run_found = True
                sm.disassociate_trial_component(TrialComponentName=component_name, TrialName=trial_name)
                try:
                    sm.delete_trial_component(TrialComponentName=component_name)
                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)
                    # component is associated with another trial
                    print("\nCould not remove this trial run. Maybe it is associated with another trial?")
                    continue
                print(f"\nTrial run <{run_name}> deleted")

    if not run_found:
        print("\nCould not find any trial run with this name in this experiment.")


def delete_experiment(sm, experiment_name):
    """Delete a SageMaker experiment and all trials associated with it.

    Parameters
    ----------
    sm : boto3 SageMaker session
        A boto3 SageMaker session created with boto3.Session().client('sagemaker').
    experiment_name : str
        The experiment name to remove trial runs from.
    """
    trials = sm.list_trials(ExperimentName=experiment_name)["TrialSummaries"]
    print("TrialNames:")
    for trial in trials:
        trial_name = trial["TrialName"]
        print(f"\n{trial_name}")

        components_in_trial = sm.list_trial_components(TrialName=trial_name)
        print("\tTrialComponentNames:")
        for component in components_in_trial["TrialComponentSummaries"]:
            component_name = component["TrialComponentName"]
            print(f"\t{component_name}")
            sm.disassociate_trial_component(TrialComponentName=component_name, TrialName=trial_name)
            try:
                sm.delete_trial_component(TrialComponentName=component_name)
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                # component is associated with another trial
                continue
            # to prevent throttling
            time.sleep(0.5)
        sm.delete_trial(TrialName=trial_name)
    sm.delete_experiment(ExperimentName=experiment_name)


def authenticated_translate_client():
    """Instantiate the Google Translate Client.

    Credentials are resolved by :func:`src.config.google_service_account_info`.

    Returns
    -------
    Client
        A Google Translate Client.
    """
    return translate.Client.from_service_account_info(google_service_account_info())
