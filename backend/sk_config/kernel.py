from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
import os
from dotenv import load_dotenv
from pathlib import Path
from openai import AsyncOpenAI
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from semantic_kernel.functions import kernel_function

load_dotenv()

def get_kernel_and_plugins():
    # Debug environment variables
    print("Checking environment variables:")
    print(f"AZURE_OPENAI_API_KEY exists: {bool(os.getenv('AZURE_OPENAI_API_KEY'))}")
    print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: {os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')}")
    print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION')}")

    # Initialize the kernel
    kernel = Kernel()
    load_dotenv()
    client = AsyncOpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"), 
    base_url="https://models.inference.ai.azure.com/",
    )


    try:
        # Configure AI service
        service_id = "chat-gpt"
        if os.getenv("AZURE_OPENAI_API_KEY"):
            # Use Azure OpenAI
            kernel.add_service(
                OpenAIChatCompletion(
                    service_id= service_id,
                    ai_model_id="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    async_client=client,
                )
            )
        else:
            # Use OpenAI
            kernel.add_service(
                OpenAIChatCompletion(
                    service_id=service_id,
                    model="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            )

        # Configure execution settings
        settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
        settings.max_tokens = 2000
        settings.temperature = 0.7
        settings.top_p = 0.8
        
        print("✅ Chat service added successfully")
    except Exception as e:
        print(f"❌ Error adding chat service: {str(e)}")
        raise e

    # Import semantic plugins
    plugins = {}
    plugin_dirs = ["goal_interpreter", "task_breakdown", "timeline_generator"]
    
    # Get the absolute path to the plugins directory
    current_dir = Path(__file__).parent
    plugins_base_dir = current_dir / "plugins"
    print(f"Loading plugins from directory: {plugins_base_dir}")
    
    for plugin_name in plugin_dirs:
        try:
            plugin_dir = str(plugins_base_dir / plugin_name)
            print(f"Attempting to load plugin from: {plugin_dir}")
            if not os.path.exists(plugin_dir):
                print(f"❌ Plugin directory does not exist: {plugin_dir}")
                continue
                
            prompt_template_config = PromptTemplateConfig(
                template=open(os.path.join(plugin_dir, "skprompt.txt")).read(),
                name=plugin_name,
                template_format="semantic-kernel",
                execution_settings=settings
            )
            
            plugins[plugin_name] = kernel.add_function(
                plugin_name=plugin_name,
                function_name=plugin_name,
                prompt_template_config=prompt_template_config
            )
            print(f"✅ Loaded plugin: {plugin_name}")
        except Exception as e:
            print(f"❌ Error loading plugin {plugin_name}: {str(e)}")
            raise e

    if not plugins:
        raise Exception("No plugins were loaded successfully")

    return kernel, plugins