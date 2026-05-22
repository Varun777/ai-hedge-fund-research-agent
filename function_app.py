import azure.functions as func
import logging
from daily_agent import generate_report

app = func.FunctionApp()

@app.timer_trigger(
    schedule="0 0 15 * * 1-5",
    arg_name="myTimer",
    run_on_startup=False,
    use_monitor=True
)
def daily_investment_memo(myTimer: func.TimerRequest) -> None:
    logging.info("Running daily investment memo agent.")
    generate_report()
    logging.info("Daily investment memo completed.")