"""
Smoke tests for deployed API service.

This script performs basic health checks and prediction tests
to verify the deployed service is working correctly.
"""

import sys
import time
import argparse
from pathlib import Path

import requests
from PIL import Image
import io


def create_test_image() -> bytes:
    """
    Create a simple test image for prediction.

    Returns:
        Image bytes in JPEG format
    """
    # Create a simple test image (red square)
    img = Image.new('RGB', (224, 224), color='red')

    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return img_byte_arr.read()


def test_health_endpoint(base_url: str, timeout: int = 10) -> bool:
    """
    Test the /health endpoint.

    Args:
        base_url: Base URL of the API (e.g., http://localhost:8000)
        timeout: Request timeout in seconds

    Returns:
        True if health check passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("Test 1: Health Endpoint")
    print("=" * 60)

    try:
        url = f"{base_url}/health"
        print(f"Calling: GET {url}")

        response = requests.get(url, timeout=timeout)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            data = response.json()
            if data.get('status') in ['healthy', 'degraded'] and data.get('model_loaded'):
                print("✅ Health check PASSED")
                return True
            else:
                print("❌ Health check FAILED: Model not loaded or unhealthy status")
                return False
        else:
            print(f"❌ Health check FAILED: Expected status 200, got {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Health check FAILED: {str(e)}")
        return False


def test_prediction_endpoint(base_url: str, image_path: str = None, timeout: int = 30) -> bool:
    """
    Test the /predict endpoint.

    Args:
        base_url: Base URL of the API
        image_path: Path to test image (optional, will create one if not provided)
        timeout: Request timeout in seconds

    Returns:
        True if prediction test passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("Test 2: Prediction Endpoint")
    print("=" * 60)

    try:
        url = f"{base_url}/predict"
        print(f"Calling: POST {url}")

        # Prepare image
        if image_path and Path(image_path).exists():
            print(f"Using image: {image_path}")
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            filename = Path(image_path).name
        else:
            print("Using generated test image")
            image_bytes = create_test_image()
            filename = "test_image.jpg"

        # Make request
        files = {'file': (filename, image_bytes, 'image/jpeg')}
        response = requests.post(url, files=files, timeout=timeout)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            data = response.json()

            # Validate response format
            required_keys = ['class', 'confidence', 'dog_probability', 'cat_probability']
            if all(key in data for key in required_keys):
                # Validate values
                if data['class'] in ['cat', 'dog']:
                    if 0.0 <= data['confidence'] <= 1.0:
                        print("✅ Prediction test PASSED")
                        print(f"   Predicted class: {data['class']}")
                        print(f"   Confidence: {data['confidence']:.4f}")
                        return True
                    else:
                        print(f"❌ Prediction test FAILED: Invalid confidence value {data['confidence']}")
                        return False
                else:
                    print(f"❌ Prediction test FAILED: Invalid class '{data['class']}'")
                    return False
            else:
                print(f"❌ Prediction test FAILED: Missing required keys in response")
                return False
        else:
            print(f"❌ Prediction test FAILED: Expected status 200, got {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction test FAILED: {str(e)}")
        return False


def wait_for_service(base_url: str, max_retries: int = 30, retry_delay: int = 2) -> bool:
    """
    Wait for the service to become available.

    Args:
        base_url: Base URL of the API
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        True if service is available, False if timeout
    """
    print("\n" + "=" * 60)
    print("Waiting for service to be ready...")
    print("=" * 60)

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Service is ready! (attempt {attempt}/{max_retries})")
                return True
        except requests.exceptions.RequestException:
            pass

        print(f"Attempt {attempt}/{max_retries}: Service not ready yet, retrying in {retry_delay}s...")
        time.sleep(retry_delay)

    print(f"❌ Service did not become ready after {max_retries} attempts")
    return False


def run_smoke_tests(
    base_url: str,
    image_path: str = None,
    wait: bool = True,
    timeout: int = 30
) -> int:
    """
    Run all smoke tests.

    Args:
        base_url: Base URL of the API
        image_path: Path to test image (optional)
        wait: Whether to wait for service to be ready
        timeout: Request timeout in seconds

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("\n" + "=" * 60)
    print("SMOKE TESTS FOR CATS vs DOGS CLASSIFIER")
    print("=" * 60)
    print(f"Target URL: {base_url}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Wait for service if requested
    if wait:
        if not wait_for_service(base_url):
            return 1

    # Run tests
    tests_passed = 0
    tests_failed = 0

    # Test 1: Health endpoint
    if test_health_endpoint(base_url, timeout):
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 2: Prediction endpoint
    if test_prediction_endpoint(base_url, image_path, timeout):
        tests_passed += 1
    else:
        tests_failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/2")
    print(f"Tests Failed: {tests_failed}/2")

    if tests_failed == 0:
        print("\n✅ All smoke tests PASSED!")
        return 0
    else:
        print(f"\n❌ {tests_failed} smoke test(s) FAILED!")
        return 1


def main():
    """Main entry point for smoke tests."""
    parser = argparse.ArgumentParser(
        description='Run smoke tests for Cats vs Dogs Classifier API'
    )
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8000',
        help='Base URL of the API (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to test image (optional)'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        help='Do not wait for service to be ready'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )

    args = parser.parse_args()

    # Run smoke tests
    exit_code = run_smoke_tests(
        base_url=args.url,
        image_path=args.image,
        wait=not args.no_wait,
        timeout=args.timeout
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
