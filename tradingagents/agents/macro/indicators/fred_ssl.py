from __future__ import annotations
import logging
import os
from functools import lru_cache
logger = logging.getLogger(__name__)

def is_cert_verify_error(exc: Exception) -> bool:
    current: Exception | None = exc
    for _ in range(6):
        if current is None:
            break
        text = str(current).upper()
        if 'CERTIFICATE_VERIFY_FAILED' in text or 'SSLCERTVERIFICATIONERROR' in text or 'UNABLE TO GET LOCAL ISSUER CERTIFICATE' in text:
            return True
        current = getattr(current, '__cause__', None) or getattr(current, '__context__', None)
    return False

@lru_cache(maxsize=1)
def bootstrap_fred_ssl() -> bool:
    if os.getenv('TRADINGAGENTS_SKIP_SSL_BOOTSTRAP', '0') == '1':
        return False
    try:
        import certifi
        ca_file = certifi.where()
        if not ca_file:
            return False
        changed = False
        if not os.getenv('SSL_CERT_FILE'):
            os.environ['SSL_CERT_FILE'] = ca_file
            changed = True
        if not os.getenv('REQUESTS_CA_BUNDLE'):
            os.environ['REQUESTS_CA_BUNDLE'] = ca_file
            changed = True
        if changed:
            logger.info('Configured SSL certificate bundle from certifi for FRED access.')
        return changed
    except Exception as exc:
        logger.debug('Unable to bootstrap certifi CA bundle: %s', exc)
        return False
