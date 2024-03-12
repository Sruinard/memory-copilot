import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)


def get_kernel(plugin_directory):
    kernel = sk.Kernel()
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

    service_id = "default"
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            deployment_name=deployment,
            endpoint=endpoint,
            api_key=api_key,
        ),
    )
    embedding_gen = AzureTextEmbedding(
        deployment_name="text-embedding-ada-002", endpoint=endpoint, api_key=api_key
    )
    kernel.add_service(embedding_gen)
    kernel.import_plugin_from_prompt_directory(plugin_directory, "SummaryPlugin")
    return kernel
