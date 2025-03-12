import azure.core.credentials
import azure.core.credentials_async


class MockAzureCredential(azure.core.credentials_async.AsyncTokenCredential):
    async def get_token(self, *scopes, **kwargs):
        return azure.core.credentials.AccessToken(
            token="mock_token",
            expires_on=1703462735,
        )
