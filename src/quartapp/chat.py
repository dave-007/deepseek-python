import json
import os

import httpx
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import AzureDeveloperCliCredential, ManagedIdentityCredential, get_bearer_token_provider
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from quart import (
    Blueprint,
    Response,
    current_app,
    render_template,
    request,
    stream_with_context,
)

bp = Blueprint("chat", __name__, template_folder="templates", static_folder="static")


@bp.before_app_serving
async def configure_openai():
    if azure_openai_key := os.getenv("AZURE_OPENAI_API_KEY_FOR_APP"):
        # use key credential
        current_app.logger.info("Using Azure OpenAI with API key")
        bp.azure_credential = AzureKeyCredential(azure_openai_key)
    elif os.getenv("RUNNING_IN_PRODUCTION"):
        client_id = os.environ["AZURE_CLIENT_ID"]
        current_app.logger.info("Using Azure OpenAI with managed identity credential for client ID: %s", client_id)
        bp.azure_credential = ManagedIdentityCredential(client_id=client_id)
    else:
        tenant_id = os.environ["AZURE_TENANT_ID"]
        current_app.logger.info("Using Azure OpenAI with Azure Developer CLI credential for tenant ID: %s", tenant_id)
        bp.azure_credential = AzureDeveloperCliCredential(tenant_id=tenant_id)

    # Get the token provider for Azure OpenAI based on the selected Azure credential
    openai_token_provider = get_bearer_token_provider(
        bp.azure_credential, "https://cognitiveservices.azure.com/.default"
    )

    class TokenBasedAuth(httpx.Auth):
        async def async_auth_flow(self, request):
            token = await openai_token_provider()
            request.headers["Authorization"] = f"Bearer {token}"
            yield request

        def sync_auth_flow(self, request):
            raise RuntimeError("Cannot use a sync authentication class with httpx.AsyncClient")

    # Create the Asynchronous Azure OpenAI client
    bp.openai_client = AsyncOpenAI(
        base_url=os.environ["AZURE_INFERENCE_ENDPOINT"],
        api_key="placeholder",
        default_query={"api-version": "preview"},
        http_client=DefaultAsyncHttpxClient(auth=TokenBasedAuth()),
    )

    # Set the model name to the Azure OpenAI model deployment name
    bp.openai_model = os.getenv("AZURE_DEEPSEEK_DEPLOYMENT")


@bp.after_app_serving
async def shutdown_openai():
    await bp.openai_client.close()


@bp.get("/")
async def index():
    return await render_template("index.html")


@bp.post("/chat/stream")
async def chat_handler():
    request_messages = (await request.get_json())["messages"]

    @stream_with_context
    async def response_stream():
        # This sends all messages, so API request may exceed token limits
        all_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ] + request_messages

        chat_coroutine = bp.openai_client.chat.completions.create(
            # Azure Open AI takes the deployment name as the model name
            model=bp.openai_model,
            messages=all_messages,
            stream=True,
        )

        try:
            async for update in await chat_coroutine:
                if update.choices:
                    yield update.choices[0].model_dump_json() + "\n"
        except Exception as e:
            current_app.logger.error(e)
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

    return Response(response_stream())
