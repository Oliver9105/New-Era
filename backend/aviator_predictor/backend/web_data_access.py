"""
Feature 2: Web Data Access
Advanced web scraping and data collection for aviator platforms
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class WebDataCollector:
    def __init__(self):
        self.active = False
        self.data_sources = {
            'bc_game': {
                'url': 'https://bc.game/game/aviator',
                'type': 'dynamic',
                'selectors': {
                    'multiplier': '[data-testid="multiplier"]',
                    'history': '[data-testid="history-item"]',
                    'current_round': '[data-testid="current-round"]'
                }
            },
            'stake': {
                'url': 'https://stake.com/casino/games/aviator',
                'type': 'dynamic',
                'selectors': {
                    'multiplier': '.multiplier-display',
                    'history': '.round-history',
                    'betting_info': '.betting-panel'
                }
            },
            'roobet': {
                'url': 'https://roobet.com/game/aviator',
                'type': 'dynamic',
                'selectors': {
                    'multiplier': '.game-multiplier',
                    'rounds': '.round-item',
                    'stats': '.stats-panel'
                }
            },
            'spribe_demo': {
                'url': 'https://demo.spribe.co/game/aviator',
                'type': 'dynamic',
                'selectors': {
                    'multiplier': '.multiplier',
                    'coefficient': '.coefficient',
                    'history': '.history-block'
                }
            }
        }
        self.scraped_data = []
        self.session = requests.Session()
        self.driver = None
        self.executor = ThreadPoolExecutor(max_workers=5)

    def initialize(self):
        """Initialize the web data collector"""
        self.active = True
        self._setup_selenium()
        logger.info("Web Data Collector initialized")

    def is_active(self):
        return self.active

    def _setup_selenium(self):
        """Setup Selenium WebDriver with optimal configuration"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")

    def collect_data(self, source: str = 'auto') -> Dict:
        """Collect data from specified source or all sources"""
        if source == 'auto':
            return self._collect_from_all_sources()
        elif source in self.data_sources:
            return self._collect_from_source(source)
        else:
            return {'error': f'Unknown source: {source}'}

    def _collect_from_all_sources(self) -> Dict:
        """Collect data from all available sources"""
        results = {}
        
        for source_name in self.data_sources.keys():
            try:
                result = self._collect_from_source(source_name)
                results[source_name] = result
                time.sleep(2)  # Rate limiting
            except Exception as e:
                results[source_name] = {'error': str(e)}
                logger.error(f"Error collecting from {source_name}: {e}")

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'sources': results,
            'successful_sources': len([r for r in results.values() if 'error' not in r])
        }

    def _collect_from_source(self, source: str) -> Dict:
        """Collect data from a specific source"""
        source_config = self.data_sources[source]
        
        if source_config['type'] == 'dynamic':
            return self._scrape_dynamic_content(source, source_config)
        else:
            return self._scrape_static_content(source, source_config)

    def _scrape_dynamic_content(self, source: str, config: Dict) -> Dict:
        """Scrape dynamic content using Selenium"""
        if not self.driver:
            return {'error': 'WebDriver not initialized'}

        try:
            logger.info(f"Scraping dynamic content from {source}")
            self.driver.get(config['url'])
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            
            # Additional wait for game to load
            time.sleep(5)
            
            data = {
                'source': source,
                'url': config['url'],
                'timestamp': datetime.utcnow().isoformat(),
                'data': {}
            }

            # Extract multiplier data
            try:
                multiplier_elements = self.driver.find_elements(By.CSS_SELECTOR, config['selectors'].get('multiplier', '.multiplier'))
                if multiplier_elements:
                    multiplier_text = multiplier_elements[0].text
                    multiplier = self._extract_multiplier_value(multiplier_text)
                    data['data']['current_multiplier'] = multiplier
            except Exception as e:
                logger.warning(f"Could not extract multiplier from {source}: {e}")

            # Extract history data
            try:
                history_elements = self.driver.find_elements(By.CSS_SELECTOR, config['selectors'].get('history', '.history'))
                history = []
                for element in history_elements[:20]:  # Last 20 rounds
                    text = element.text
                    multiplier = self._extract_multiplier_value(text)
                    if multiplier:
                        history.append(multiplier)
                data['data']['history'] = history
            except Exception as e:
                logger.warning(f"Could not extract history from {source}: {e}")

            # Extract additional game data
            try:
                page_source = self.driver.page_source
                game_data = self._extract_game_data_from_html(page_source)
                data['data'].update(game_data)
            except Exception as e:
                logger.warning(f"Could not extract additional data from {source}: {e}")

            return data

        except Exception as e:
            logger.error(f"Error scraping {source}: {e}")
            return {'error': str(e), 'source': source}

    def _scrape_static_content(self, source: str, config: Dict) -> Dict:
        """Scrape static content using requests"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = self.session.get(config['url'], headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = {
                'source': source,
                'url': config['url'],
                'timestamp': datetime.utcnow().isoformat(),
                'data': {}
            }

            # Extract data based on selectors
            for data_type, selector in config['selectors'].items():
                elements = soup.select(selector)
                if elements:
                    data['data'][data_type] = [elem.get_text(strip=True) for elem in elements]

            return data

        except Exception as e:
            logger.error(f"Error scraping static content from {source}: {e}")
            return {'error': str(e), 'source': source}

    def _extract_multiplier_value(self, text: str) -> float:
        """Extract numerical multiplier value from text"""
        try:
            # Look for patterns like "1.23x", "1.23", "x1.23"
            patterns = [
                r'(\d+\.?\d*)x',  # 1.23x
                r'x(\d+\.?\d*)',  # x1.23
                r'(\d+\.\d+)',    # 1.23
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text.replace(',', '.'))
                if match:
                    return float(match.group(1))
            
            return None
        except Exception:
            return None

    def _extract_game_data_from_html(self, html: str) -> Dict:
        """Extract game data from HTML source"""
        data = {}
        
        try:
            # Look for JSON data in script tags
            soup = BeautifulSoup(html, 'html.parser')
            scripts = soup.find_all('script')
            
            for script in scripts:
                if script.string:
                    # Look for game state data
                    if 'gameState' in script.string or 'aviator' in script.string.lower():
                        try:
                            # Try to extract JSON objects
                            json_matches = re.findall(r'\{[^{}]*\}', script.string)
                            for match in json_matches:
                                try:
                                    parsed = json.loads(match)
                                    if isinstance(parsed, dict) and any(key in str(parsed).lower() for key in ['multiplier', 'coefficient', 'round']):
                                        data['game_state'] = parsed
                                        break
                                except json.JSONDecodeError:
                                    continue
                        except Exception:
                            continue

            # Extract meta information
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                if meta.get('name') == 'description' or meta.get('property') == 'og:description':
                    data['description'] = meta.get('content', '')

        except Exception as e:
            logger.warning(f"Error extracting game data from HTML: {e}")

        return data

    def get_sources(self) -> Dict:
        """Get available data sources"""
        return {
            'sources': self.data_sources,
            'total_sources': len(self.data_sources),
            'active_sources': len([s for s in self.data_sources.values() if s.get('status', 'unknown') != 'disabled'])
        }

    def get_realtime_data(self) -> Dict:
        """Get real-time data for streaming"""
        # For demo purposes, return simulated real-time data
        import random
        
        current_multiplier = round(random.uniform(1.0, 10.0), 2)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'current_multiplier': current_multiplier,
            'round_id': int(time.time()),
            'status': 'active' if current_multiplier < 5.0 else 'crashed',
            'players_count': random.randint(50, 500),
            'total_bet': round(random.uniform(1000, 50000), 2)
        }

    async def async_collect_data(self, sources: List[str]) -> Dict:
        """Asynchronously collect data from multiple sources"""
        async def collect_source(source):
            try:
                # Simulate async data collection
                await asyncio.sleep(1)
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._collect_from_source, source
                )
            except Exception as e:
                return {'error': str(e), 'source': source}

        tasks = [collect_source(source) for source in sources if source in self.data_sources]
        results = await asyncio.gather(*tasks)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'results': results,
            'total_sources': len(sources),
            'successful_collections': len([r for r in results if 'error' not in r])
        }

    def monitor_data_quality(self) -> Dict:
        """Monitor the quality of collected data"""
        recent_data = self.scraped_data[-100:] if len(self.scraped_data) > 100 else self.scraped_data
        
        quality_metrics = {
            'total_samples': len(recent_data),
            'valid_multipliers': 0,
            'invalid_multipliers': 0,
            'average_multiplier': 0,
            'data_sources_active': 0,
            'last_collection': None
        }

        if recent_data:
            valid_multipliers = []
            for sample in recent_data:
                if 'data' in sample and 'current_multiplier' in sample['data']:
                    multiplier = sample['data']['current_multiplier']
                    if isinstance(multiplier, (int, float)) and 1.0 <= multiplier <= 100.0:
                        valid_multipliers.append(multiplier)
                        quality_metrics['valid_multipliers'] += 1
                    else:
                        quality_metrics['invalid_multipliers'] += 1

            if valid_multipliers:
                quality_metrics['average_multiplier'] = sum(valid_multipliers) / len(valid_multipliers)
            
            quality_metrics['last_collection'] = recent_data[-1].get('timestamp')

        # Check active sources
        for source in self.data_sources:
            try:
                # Quick connectivity test
                result = self._collect_from_source(source)
                if 'error' not in result:
                    quality_metrics['data_sources_active'] += 1
            except Exception:
                pass

        quality_metrics['data_quality_score'] = (
            quality_metrics['valid_multipliers'] / max(quality_metrics['total_samples'], 1)
        ) * 100

        return quality_metrics

    def export_collected_data(self, format: str = 'json') -> Dict:
        """Export collected data in specified format"""
        if format == 'json':
            return {
                'export_format': 'json',
                'timestamp': datetime.utcnow().isoformat(),
                'total_records': len(self.scraped_data),
                'data': self.scraped_data
            }
        elif format == 'csv':
            # Convert to CSV format
            import pandas as pd
            try:
                df = pd.DataFrame(self.scraped_data)
                csv_data = df.to_csv(index=False)
                return {
                    'export_format': 'csv',
                    'timestamp': datetime.utcnow().isoformat(),
                    'total_records': len(self.scraped_data),
                    'data': csv_data
                }
            except Exception as e:
                return {'error': f'CSV export failed: {e}'}
        else:
            return {'error': f'Unsupported format: {format}'}

    def __del__(self):
        """Cleanup resources"""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
