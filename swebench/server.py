from flask import Flask, request, jsonify
from pathlib import Path
import json
import logging

from swebench.harness.run_evaluation import run_instance, make_test_spec
from swebench.harness.docker_utils import should_remove
import docker

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.get_json()
        
        # Required fields
        test_spec = make_test_spec(data['instance'])
        prediction = {
            'instance_id': test_spec.instance_id,
            'prediction': data.get('prediction', ''),
            'model': data.get('model', 'api-user')
        }
        
        # Optional parameters with defaults
        rm_image = data.get('rm_image', True) 
        force_rebuild = data.get('force_rebuild', False)
        run_id = data.get('run_id', 'api-run')
        timeout = data.get('timeout', 1800)  # 30 minutes default

        # Initialize Docker client
        client = docker.from_env()
        
        # Run evaluation
        result = run_instance(
            test_spec=test_spec,
            pred=prediction,
            rm_image=rm_image,
            force_rebuild=force_rebuild,
            client=client,
            run_id=run_id,
            timeout=timeout
        )
        
        if result is None:
            return jsonify({'error': 'Evaluation failed'}), 500
            
        instance_id, report = result
        return jsonify(report)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
