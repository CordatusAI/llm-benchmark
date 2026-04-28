import asyncio
import io
import time
import zipfile
import requests
import statistics
import os
import httpx
import random
import logging
import math

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_KEY = "Dummy"
TO_MS = 1000
DEFAULT_BASE_URL = "http://127.1.1.1:8000/v1"
DEFAULT_MAX_TOKENS = 128
DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4, 8]
MIN_ROUNDS = 10
DEFAULT_TTFT_SLO = 1000
DEFAULT_TPS_SLO = 15
DEFAULT_THINK_TIME = 45
OUTPUT_DIR = "out"
PROMPTS_FILE = "prompts.txt"
LOGO_PATH = "images/CORDATUS_LOGO.png"
REQUEST_TIMEOUT = 50

def get_model_list(url: str, api_key:str= API_KEY) -> List[str]:
    """
    Returns a list of model IDs from an OpenAI compatible server.
    The model list is dynamic and not cached.

    Args:
        url: API base URL

    Returns:
        List of model IDs

    Raises:
        requests.exceptions.RequestException: If API request fails
    """
    try:
        response = requests.get(
            f"{url}/models",
            timeout=REQUEST_TIMEOUT,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
             verify=False
        )
        response.raise_for_status()
        data = response.json()
        model_ids = [model['id'] for model in data.get('data', [])]

        if not model_ids:
            logger.warning("Could not retrieve model list from server.")

        return model_ids

    except requests.exceptions.Timeout:
        logger.warning(f"Server timed out ({REQUEST_TIMEOUT}s)")
        return []
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to server. Check URL.")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"API Error: {str(e)}")
        return []
    except (ValueError, KeyError) as e:
        logger.error(f"Error parsing response: {str(e)}")
        return []



class LLMBenchmark:
    """
    A class for benchmarking LLM models with various performance metrics.
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        """
        Initialize the benchmark client.

        Args:
            base_url: The base URL of the API server
            api_key: The API key for authentication
            model: The model name to benchmark
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model

        # Disable SSL verification (security risk — only use in test/dev environments)
        http_client = httpx.Client(
            verify=False,
        )

        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=api_key if api_key else None,
            timeout=httpx.Timeout(60.0, connect=10.0, read=30.0, write=10.0),
            max_retries=3,
            http_client=http_client
        )

    def measure_single(self, prompt: str, max_tokens: int) -> Dict:
        """
        Run a completion request and measure timing metrics.

        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary containing performance metrics
        """
        start_time = time.time()
        first_token_time = None
        last_token_time = None
        tokens = 0
        total_tokens = 0
        total_content = []
        chunk_count = 0

        try:
            # Streaming completion with explicit timeout
            logger.info(f"Starting completion for model: {self.model}")

            # Create stream with timeout
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True}
            )

            # Process stream with chunk timeout protection
            last_chunk_time = time.time()
            chunk_timeout = 100

            for event in stream:
                current_time = time.time()

                if current_time - last_chunk_time > chunk_timeout:
                    logger.error("Chunk timeout reached")
                    break

                last_chunk_time = current_time

                # Process the event
                if event.choices:
                    delta = event.choices[0].delta

                    # Check for content (both regular and reasoning)
                    content = None
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                    elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        content = delta.reasoning_content
                    elif hasattr(delta, 'reasoning') and delta.reasoning:
                        content = delta.reasoning
                    if content:
                        chunk_count += 1
                        now = time.time()

                        if first_token_time is None:
                            first_token_time = now
                            logger.info(f"First token received after {(first_token_time - start_time)*1000:.2f}ms")

                        last_token_time = now
                        total_content.append(content)

                # Get usage information if available
                if hasattr(event, 'usage') and event.usage is not None:
                    total_tokens = event.usage.completion_tokens
                    logger.debug(f"Token count: {total_tokens}")

            logger.info(f"Stream completed. Chunks: {chunk_count}, Tokens: {total_tokens}")

        except httpx.ReadTimeout:
            logger.error("Read timeout occurred")
            return {"error": "Read timeout - server took too long to respond"}
        except httpx.WriteTimeout:
            logger.error("Write timeout occurred")
            return {"error": "Write timeout - failed to send request"}
        except httpx.ConnectTimeout:
            logger.error("Connection timeout occurred")
            return {"error": "Connection timeout - could not connect to server"}
        except Exception as e:
            logger.error(f"Error during completion: {type(e).__name__}: {e}")
            return {"error": f"{type(e).__name__}: {str(e)}"}

        end_time = time.time()

        # Calculate metrics
        if not total_content:
            logger.warning("No content received")
            return {"error": "No content received"}

        # Use actual token count if available, otherwise estimate from chunks
        tokens = total_tokens if total_tokens > 0 else chunk_count

        # Calculate timing metrics
        ttft = (first_token_time - start_time) * TO_MS if first_token_time else 0
        total_time = end_time - start_time
        latency = total_time
        itl = ((last_token_time - first_token_time) * TO_MS / max(tokens - 1, 1)) if tokens > 1 else 0
        tps = tokens / total_time if total_time > 0 else 0

        content = "".join(total_content)

        result = {
            "ttft": ttft,
            "itl": itl,
            "tps": tps,
            "latency": latency,
            "tokens": tokens,
            "content": content,
        }

        logger.info(f"Request completed - TTFT: {ttft:.2f}ms, TPS: {tps:.2f}, Tokens: {tokens}")
        return result

    async def run_concurrent(self, prompts: List[str], max_tokens: int, 
                           concurrency: int, progress_callback) -> List[Dict]:
        """
        Run benchmark with concurrent requests.

        Args:
            prompts: List of prompts to test
            max_tokens: Maximum tokens per response
            concurrency: Number of concurrent requests
            progress_callback: Callback function to update progress

        Returns:
            List of benchmark results
        """
        loop = asyncio.get_running_loop()
        results = []

        # Add timeout to concurrent execution
        async with asyncio.timeout(300):  # 5 minutes total timeout
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    loop.run_in_executor(
                        executor, 
                        self.measure_single, 
                        prompt, 
                        max_tokens
                    ) 
                    for prompt in prompts
                ]

                completed_count = 0
                for coro in asyncio.as_completed(futures):
                    try:
                        result = await coro
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Future failed: {e}")
                        results.append({"error": str(e)})
                    finally:
                        completed_count += 1
                        progress_callback(completed_count, len(prompts))

        return results

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate statistical metrics from benchmark results.

        Args:
            results: List of benchmark results

        Returns:
            Dictionary containing calculated metrics
        """
        valid_results = [
            r for r in results 
            if 'error' not in r and r['ttft'] is not None
        ]

        if not valid_results:
            return {"error": "No valid results"}

        # Extract metric values
        ttfts = [r['ttft'] for r in valid_results]
        itls = [r['itl'] for r in valid_results]
        tps_list = [r['tps'] for r in valid_results]
        latencies = [r['latency'] for r in valid_results]

        # Calculate throughput
        throughput = len(valid_results) / sum(latencies) if latencies else 0

        def calculate_percentiles(data: List[float]) -> Dict[str, float]:
            """Calculate mean, p50, and p90 percentiles."""
            if not data:
                return {"mean": 0, "p50": 0, "p90": 0}

            sorted_data = sorted(data)
            return {
                "mean": statistics.mean(data),
                "p50": statistics.median(data),
                "p90": sorted_data[min(int(math.ceil(len(sorted_data) * 0.9)) - 1, len(sorted_data) - 1)] if sorted_data else 0
            }

        return {
            "TTFT": calculate_percentiles(ttfts),
            "ITL": calculate_percentiles(itls),
            "TPS": calculate_percentiles(tps_list),
            "Latency": calculate_percentiles(latencies),
            "Throughput (RPS)": throughput
        }

    def plot_results(self, all_metrics: Dict) -> Dict[str, go.Figure]:
        """
        Create visualization plots for benchmark results.

        Args:
            all_metrics: Dictionary of metrics for each concurrency level

        Returns:
            Dictionary of Plotly figures
        """
        figures = {}
        metrics_to_plot = ["TTFT", "ITL", "TPS", "Latency"]
        concurrency_levels = list(all_metrics.keys())

        for metric in metrics_to_plot:
            means = [all_metrics[conc][metric]["mean"] for conc in concurrency_levels]
            p90s = [all_metrics[conc][metric]["p90"] for conc in concurrency_levels]

            trace_mean = go.Bar(
                x=[str(conc) for conc in concurrency_levels],
                y=means,
                name="Mean",
                marker_color="skyblue",
                width=0.3
            )
            trace_p90 = go.Bar(
                x=[str(conc) for conc in concurrency_levels],
                y=p90s,
                name="P90",
                marker_color="salmon",
                width=0.3
            )

            fig = go.Figure(data=[trace_mean, trace_p90])

            # Set appropriate units
            metric_units = {
                "TPS": "tokens/s",
                "TTFT": "ms",
                "ITL": "ms",
                "Latency": "s"
            }
            metric_unit = metric_units.get(metric, "")

            fig.update_layout(
                title=f"{metric} Across Concurrency Levels",
                xaxis_title="Concurrency Level",
                yaxis_title=f"{metric} ({metric_unit})",
                barmode="group",
                template="plotly_white"
            )

            figures[metric] = fig

        # Plot throughput separately
        throughputs = [all_metrics[conc]["Throughput (RPS)"] for conc in concurrency_levels]
        fig = go.Figure(data=[
            go.Bar(
                x=[str(conc) for conc in concurrency_levels],
                y=throughputs,
                name="Throughput",
                marker_color="lightblue",
                width=0.25
            )
        ])
        fig.update_layout(
            title="Throughput Across Concurrency Levels",
            xaxis_title="Concurrency Level",
            yaxis_title="Throughput (RPS)",
            template="plotly_white"
        )
        figures['Throughput'] = fig

        return figures

    async def run(self, prompts: List[str], max_tokens: int,
                  concurrency_levels: List[int]) -> Dict:
        """
        Run the complete benchmark suite.

        Args:
            prompts: List of prompts to test
            max_tokens: Maximum tokens per response
            concurrency_levels: List of concurrency levels to test

        Returns:
            Dictionary containing all benchmark results
        """
        st.write("Warm Up Started ... ")
        self.measure_single("What is warmup?", 32)
        st.write("Warm Up Ended ...")

        all_metrics = {}
        total_tasks = sum(MIN_ROUNDS * conc for conc in concurrency_levels)
        progress_bar = st.progress(0)
        completed_tasks = 0

        def update_progress(current: int, total: int) -> None:
            nonlocal completed_tasks
            completed_tasks += 1
            progress = min(completed_tasks / total_tasks, 1.0)
            progress_bar.progress(progress)

        for conc in concurrency_levels:
            needed = MIN_ROUNDS * conc
            if needed <= len(prompts):
                batch = prompts[:needed]
            else:
                batch = [prompts[i % len(prompts)] for i in range(needed)]

            st.write(f"Running with concurrency level: {conc} ({len(batch)} prompts, {len(batch)//conc} rounds)")

            try:
                results = await self.run_concurrent(batch, max_tokens, conc, update_progress)
                metrics = self.calculate_metrics(results)

                if "error" in metrics:
                    st.error(f"Error in metrics calculation: {metrics['error']}")
                    continue

                all_metrics[conc] = metrics
                st.write(f"Metrics for concurrency level {conc}:")
                st.json(metrics)

            except Exception as e:
                logger.error(f"Error running benchmark with concurrency {conc}: {e}")
                st.error(f"Error with concurrency {conc}: {e}")
                continue

        return all_metrics


def load_prompts() -> List[str]:
    """
    Load all prompts from file.

    Returns:
        List of prompt strings
    """
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            prompts = f.readlines()
            prompts = [p.strip() for p in prompts if p.strip()]
            random.shuffle(prompts)
            return prompts

    except FileNotFoundError:
        st.error(f"Error: {PROMPTS_FILE} file not found!")
        return []
    except Exception as e:
        st.error(f"Error loading prompts: {e}")
        return []


def build_zip(model_name: str, df: pd.DataFrame, figures: Dict[str, go.Figure]) -> bytes:
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        zip_file.writestr(f"{model_name}-table.csv", csv_buffer.getvalue())
        
        for metric, fig in figures.items():
            try:
                png_buffer = io.BytesIO()
                fig.write_image(png_buffer, format='png')
                zip_file.writestr(f"{model_name}-{metric}.png", png_buffer.getvalue())
                
                html_buffer = io.StringIO()
                fig.write_html(html_buffer)
                zip_file.writestr(f"{model_name}-{metric}.html", html_buffer.getvalue())
            except Exception as e:
                logger.error(f"Error saving figure {metric}: {e}")
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def calculate_capacity(all_metrics: Dict, ttft_slo: int, tps_slo: int,
                       percentile_key: str, think_time: int) -> Dict:
    c_max = None
    slo_status = []

    for conc in sorted(all_metrics.keys()):
        metrics = all_metrics[conc]
        if "error" in metrics:
            continue

        ttft_val = metrics["TTFT"][percentile_key]
        tps_val = metrics["TPS"][percentile_key]

        ttft_ok = ttft_val <= ttft_slo
        tps_ok = tps_val >= tps_slo

        slo_status.append({
            "conc": conc,
            "ttft": ttft_val,
            "tps": tps_val,
            "ttft_ok": ttft_ok,
            "tps_ok": tps_ok,
        })

        if ttft_ok and tps_ok:
            c_max = conc

    latency_val = all_metrics[c_max]["Latency"][percentile_key] if c_max else 0
    total_users = int(c_max * (1 + think_time / latency_val)) if c_max else 0

    return {
        "c_max": c_max,
        "total_users": total_users,
        "latency": latency_val,
        "slo_status": slo_status
    }


def create_results_dataframe(all_metrics: Dict) -> pd.DataFrame:
    """
    Create a DataFrame from benchmark results.

    Args:
        all_metrics: Dictionary of metrics for each concurrency level

    Returns:
        DataFrame with formatted results
    """
    data = []
    for conc, metrics in all_metrics.items():
        if "error" not in metrics:
            data.append({
                "Concurrency": conc,
                "TTFT (Mean, ms)": f"{metrics['TTFT']['mean']:.2f}",
                "TTFT (p50, ms)": f"{metrics['TTFT']['p50']:.2f}",
                "TTFT (p90, ms)": f"{metrics['TTFT']['p90']:.2f}",
                "ITL (Mean, ms)": f"{metrics['ITL']['mean']:.2f}",
                "ITL (p50, ms)": f"{metrics['ITL']['p50']:.2f}",
                "ITL (p90, ms)": f"{metrics['ITL']['p90']:.2f}",
                "TPS (Mean, tokens/s)": f"{metrics['TPS']['mean']:.2f}",
                "TPS (p50, tokens/s)": f"{metrics['TPS']['p50']:.2f}",
                "TPS (p90, tokens/s)": f"{metrics['TPS']['p90']:.2f}",
                "Latency (Mean, s)": f"{metrics['Latency']['mean']:.2f}",
                "Latency (p50, s)": f"{metrics['Latency']['p50']:.2f}",
                "Latency (p90, s)": f"{metrics['Latency']['p90']:.2f}",
                "Throughput (RPS)": f"{metrics['Throughput (RPS)']:.2f}"
            })
    return pd.DataFrame(data)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'models' not in st.session_state:
        st.session_state.models = []
    if 'last_base_url' not in st.session_state:
        st.session_state.last_base_url = None


def configure_page():
    """Configure page layout and styles."""

    # Add CSS styles
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                width: 400px !important;
            }
            [data-testid="stSidebar"] > div:first-child {
                width: 400px !important;
            }
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1e3a8a;
                text-align: center;
                margin-bottom: 2rem;
            }
            .subtitle {
                font-size: 1.2rem;
                color: #6b7280;
                text-align: center;
                margin-bottom: 3rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Page configuration
    st.set_page_config(
        page_title="LLM Benchmark",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )



def main() -> None:
    """Main application function."""
    initialize_session_state()

    configure_page()

    # Sidebar setup
    if os.path.exists(LOGO_PATH):
        st.sidebar.image(LOGO_PATH)
    st.sidebar.title("LLM Benchmark Tool")
    st.sidebar.write("Enter the details below to benchmark an LLM model.")

    api_key = st.sidebar.text_input("API KEY ", value=API_KEY)
    # Input controls
    base_url = st.sidebar.text_input("Base URL", value=DEFAULT_BASE_URL)

    # Handle base URL changes
    if base_url != st.session_state.last_base_url:
        st.session_state.models = []
        st.session_state.last_base_url = base_url
        st.rerun()

    # Refresh models button
    if st.sidebar.button(
        "🔄 Refresh Models",
        key="refresh_models",
        help="Refresh model list from server"
    ) and base_url:
        with st.spinner("Fetching model list..."):
            new_models = get_model_list(base_url)

        if new_models:
            st.session_state.models = new_models
            st.success(f"Successfully loaded {len(new_models)} models!")
            st.rerun()
        else:
            st.error("Failed to fetch model list!")

    # Model selection
    model = st.sidebar.selectbox("Model Name:", st.session_state.models)

    # Benchmark parameters
    concurrency_levels = st.sidebar.multiselect(
        "Concurrency Levels", 
        options=[1, 2, 4, 8, 16, 32, 64], 
        default=DEFAULT_CONCURRENCY_LEVELS
    )

    ttft_slo = DEFAULT_TTFT_SLO
    tps_slo = DEFAULT_TPS_SLO
    percentile_key = "p90"
    think_time = DEFAULT_THINK_TIME

    if 'sonuc' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Capacity Planning")
        ttft_slo = st.sidebar.number_input(
            "TTFT SLO (ms)", min_value=100, max_value=10000,
            value=DEFAULT_TTFT_SLO
        )
        tps_slo = st.sidebar.number_input(
            "TPS SLO (tok/s)", min_value=1, max_value=1000,
            value=DEFAULT_TPS_SLO
        )
        percentile_key = st.sidebar.radio(
            "Percentile", ["p90", "mean"], horizontal=True,
            format_func=lambda x: "P90" if x == "p90" else "Mean"
        )
        think_time = st.sidebar.number_input(
            "Think Time (s)", min_value=1, max_value=600,
            value=DEFAULT_THINK_TIME
        )

    if st.sidebar.button("Run Benchmark"):
        if 'sonuc' in st.session_state:
            del st.session_state.sonuc

        with st.spinner("Running benchmark..."):
            prompts = load_prompts()
            if not prompts:
                st.error("Could not load prompts. Please check the prompts.txt file.")
            else:
                try:
                    benchmark = LLMBenchmark(base_url, api_key, model)
                    all_metrics = asyncio.run(
                        benchmark.run(prompts, DEFAULT_MAX_TOKENS, concurrency_levels)
                    )

                    if all_metrics:
                        df = create_results_dataframe(all_metrics)
                        figures = benchmark.plot_results(all_metrics)
                        model_name = model.split('/')[-1]
                        st.session_state.sonuc = (model_name, df, figures, all_metrics)
                    else:
                        st.error("No benchmark results were generated.")
                except Exception as e:
                    logger.error(f"Benchmark failed: {e}")
                    st.error(f"Benchmark failed: {e}")

    if 'sonuc' in st.session_state:
        model_name, df, figures, all_metrics = st.session_state.sonuc

        st.subheader("Benchmark Results")
        st.dataframe(df, width='stretch')

        cap = calculate_capacity(all_metrics, ttft_slo, tps_slo, percentile_key, think_time)
        with st.expander("Capacity Planning", expanded=False):
            if cap["c_max"]:
                st.success(
                    f"Max Concurrency: {cap['c_max']} | "
                    f"Total Users: ~{cap['total_users']} | "
                    f"N = {cap['c_max']} x (1 + {think_time} / {cap['latency']:.2f})"
                )
            else:
                st.error("No concurrency level meets SLO thresholds!")

            cap_data = []
            for s in cap["slo_status"]:
                cap_data.append({
                    "Concurrency": s["conc"],
                    f"TTFT ({percentile_key}, ms)": f"{s['ttft']:.0f}",
                    "TTFT SLO": "OK" if s["ttft_ok"] else "X",
                    f"TPS ({percentile_key}, tok/s)": f"{s['tps']:.1f}",
                    "TPS SLO": "OK" if s["tps_ok"] else "X",
                })
            st.dataframe(pd.DataFrame(cap_data), width='stretch')

        st.subheader("Visualizations")
        for metric, fig in figures.items():
            st.plotly_chart(fig, use_container_width=True)

        if 'zip_data' not in st.session_state or st.session_state.get('zip_model') != model_name:
            st.session_state.zip_data = build_zip(model_name, df, figures)
            st.session_state.zip_model = model_name

        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="📥 Download All Results (ZIP)",
            data=st.session_state.zip_data,
            file_name=f"{model_name}_results.zip",
            mime="application/zip"
        )

        if cap["c_max"] is None:
            st.sidebar.error("No concurrency level meets SLO thresholds!")
        else:
            st.sidebar.success(f"Max Concurrency: {cap['c_max']}")
            st.sidebar.info(f"Total Users: ~{cap['total_users']}")
            st.sidebar.caption(
                f"N = {cap['c_max']} x (1 + {think_time} / {cap['latency']:.2f})"
            )

        status_lines = []
        for s in cap["slo_status"]:
            ttft_icon = "OK" if s["ttft_ok"] else "X"
            tps_icon = "OK" if s["tps_ok"] else "X"
            status_lines.append(
                f"Conc {s['conc']}: TTFT={s['ttft']:.0f}ms [{ttft_icon}] "
                f"TPS={s['tps']:.1f} [{tps_icon}]"
            )
        st.sidebar.code("\n".join(status_lines))


if __name__ == "__main__":
    main()