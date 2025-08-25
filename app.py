import streamlit as st
import os
from dotenv import load_dotenv
from portia import (
    Config,
    DefaultToolRegistry,
    LLMProvider,
    LogLevel,
    Portia,
    StorageClass,
    PlanRunState,
)
from portia.cli import CLIExecutionHooks
from portia.open_source_tools.browser_tool import BrowserTool, BrowserInfrastructureOption
from portia import open_source_tool_registry
import logging

# Load environment variables from a .env file at the start
load_dotenv()

# --- Page & Logger Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Travel Agenta"
)
logging.basicConfig(level=logging.INFO) # Added basic logging for debugging

# --- Agent Configuration ---
@st.cache_resource
def setup_portia_agent():
    """Sets up and caches the Portia agent using environment variables."""
    # Retrieve API keys from environment variables
    portia_key = os.getenv("PORTIA_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not all([portia_key, gemini_key, tavily_key]):
        st.error("Cannot initialize agent. One or more API keys are missing from the environment variables.")
        return None

    try:
        # Set environment variables required by Portia and underlying tools
        os.environ["PORTIA_API_KEY"] = portia_key
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["TAVILY_API_KEY"] = tavily_key

        my_config = Config.from_default(
            llm_provider=LLMProvider.GOOGLE,
            storage_class=StorageClass.CLOUD,
            storage_dir="./production_states",
            default_log_level=LogLevel.DEBUG,
            default_log_sink="./agent_audit.log",
            json_log_serialize=True,
        )

        portia_agent = Portia(
            config=my_config,
            tools=DefaultToolRegistry(my_config) + open_source_tool_registry + [
                BrowserTool(infrastructure_option=BrowserInfrastructureOption.REMOTE)
            ],
            execution_hooks=CLIExecutionHooks(),
        )
        return portia_agent
    except Exception as e:
        logging.error(f"Failed to initialize Portia agent: {e}")
        st.error(f"Failed to initialize the agent. Please check your API keys and configuration. Error: {e}")
        return None

# --- Main Application UI ---
st.title("🌍 TRAVEL AGENTA")

# Initialize the agent
portia = setup_portia_agent()

if not portia:
    st.info("⬅ Please ensure your API keys are set in a `.env` file to start planning your trip.")
    st.stop()


col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("Plan Your Trip")

    form_col1, form_col2 = st.columns(2)
    with form_col1:
        origin = st.text_input("FROM", "New York")
        start_date = st.text_input("START DATE", "15 Oct 2025")
        duration = st.text_input("DURATION", "7 days")
    with form_col2:
        destination = st.text_input("TO", "Paris")
        return_date = st.text_input("RETURN DATE", "22 Oct 2025")
        budget = st.text_input("BUDGET", "Approx $2000")

    st.markdown("---")
    trip_type = st.radio("TRIP TYPE", ["Solo", "Family", "Friends", "Business"], horizontal=True)
    accommodation = st.radio("ACCOMMODATION", ["Budget", "Luxury", "Flexible"], horizontal=True)
    food_preferences = st.radio("FOOD PREFERENCES", ["Veg", "Non Veg", "Flexible"], horizontal=True)
    st.markdown("---")

    activities = st.text_input("ACTIVITIES", "Museums, historic sites, local cuisine tours")
    emails_input = st.text_input("EMAILS", "example1@gmail.com, example2@gmail.com")
    custom_instructions = st.text_area("CUSTOM INSTRUCTIONS", "Focus on cultural experiences and avoid tourist traps. Include at least one day trip outside the main city.")

    if st.button("GENERATE A PLAN", use_container_width=True):
        required_fields = { "Origin": origin, "Destination": destination, "Start Date": start_date,
                            "Return Date": return_date, "Duration": duration, "Budget": budget, "Emails": emails_input }
        missing = [field for field, value in required_fields.items() if not value.strip()]

        if missing:
            st.error(f"⚠️ Please fill all required fields: {', '.join(missing)}")
        else:
            emails = [email.strip() for email in emails_input.split(',')]

            task = f"""
            Make a Travel Plan with the following details:
            - Origin: {origin} - Destination: {destination} - Start Date: {start_date} - Return Date: {return_date}
            - Duration: {duration} - Trip Type: {trip_type} - Budget: {budget} - Accommodation Preference: {accommodation}
            - Activities & Interests: {activities} - Dining Preferences: {food_preferences} - Custom Instructions: {custom_instructions}
            - Expected Final Output: A detailed day-by-day itinerary, travel options, and recommendations.
            - Once the plan is generated, email a comprehensive report to the following addresses: {", ".join(emails)}.
            - Add these events to a calendar.
            - Send calendar invitations to these emails: {", ".join(emails)}
            """

            with col2:
                st.header("AGENT STATUS")
                plan_steps_container = st.container()
                final_output_container = st.container()

                # --- Agent Planning Stage ---
                try:
                    with plan_steps_container:
                        st.subheader("PLAN STEPS")
                        with st.spinner('Agent is generating the plan...'):
                            plan = portia.plan(task)
                            plan_output = plan.pretty_print()
                            st.text_area("Generated Plan Steps", plan_output, height=300)
                            st.session_state.plan_generated = plan # Store plan for execution
                except Exception as e:
                    logging.error(f"An error occurred during planning: {e}")
                    st.error(f"An error occurred during the planning phase: {e}")
                    st.stop() # Stop execution if planning fails

                # --- Agent Execution Stage ---
                try:
                    with final_output_container:
                        st.subheader("FINAL OUTPUT")
                        with st.spinner('Agent is executing the plan and sending communications...'):
                            plan_to_run = st.session_state.get("plan_generated")
                            if plan_to_run:
                                plan_run = portia.run_plan(plan_to_run, end_user="Streamlit User")

                                # Check the final state of the agent run
                                if plan_run and plan_run.state == PlanRunState.COMPLETE:
                                    final_summary = (
                                        f"✅ Plan successfully executed!\n\nA detailed travel itinerary for your {trip_type} trip "
                                        f"from {origin} to {destination} has been generated and sent to: {emails_input}."
                                    )
                                    st.success("Execution Complete!")
                                    st.text_area("Execution Summary", final_summary, height=150)
                                else:
                                    failed_state = plan_run.state.value if plan_run else 'N/A'
                                    st.error(f"❌ Agent run failed to complete. Final state: {failed_state}")
                            else:
                                st.error("No plan was available to execute.")
                except Exception as e:
                    logging.error(f"An error occurred during execution: {e}")
                    st.error(f"An error occurred during the plan execution phase: {e}")
