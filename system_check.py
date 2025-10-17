#!/usr/bin/env python3
"""
Quick dependency check and system verification
"""

import sys
import importlib

def check_dependencies():
    """Check if all required dependencies are available"""
    
    print("🔍 Checking system dependencies...")
    
    dependencies = [
        ('flask', 'Flask'),
        ('requests', 'HTTP requests'),
        ('numpy', 'NumPy (optional, with fallback)'),
        ('pandas', 'Pandas (optional, with fallback)'),
        ('sklearn', 'Scikit-learn (optional, with fallback)'),
        ('websocket', 'WebSocket client'),
        ('threading', 'Threading support'),
        ('json', 'JSON support'),
        ('datetime', 'Date/time support'),
        ('logging', 'Logging support')
    ]
    
    results = {}
    
    for module_name, description in dependencies:
        try:
            importlib.import_module(module_name)
            results[module_name] = True
            print(f"✅ {description}: Available")
        except ImportError:
            results[module_name] = False
            print(f"❌ {description}: Missing")
    
    print("\n📊 Dependency Summary:")
    available = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Available: {available}/{total}")
    
    # Check critical dependencies
    critical = ['flask', 'requests', 'threading', 'json', 'datetime', 'logging']
    critical_available = all(results.get(dep, False) for dep in critical)
    
    if critical_available:
        print("✅ All critical dependencies available - System should work!")
    else:
        print("❌ Some critical dependencies missing")
        
    return results

def check_file_structure():
    """Check if all required files are present"""
    
    print("\n🗂️ Checking file structure...")
    
    import os
    
    required_files = [
        'app.py',
        'backend/real_data_collector.py',
        'backend/real_prediction_engine.py',
        'backend/real_network_inspector.py',
        'backend/real_game_analyzer.py',
        'templates/index.html',
        'requirements.txt'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: Present")
        else:
            print(f"❌ {file_path}: Missing")

def check_backend_methods():
    """Check if backend classes have required methods"""
    
    print("\n🔧 Checking backend implementations...")
    
    try:
        # Check RealPredictionEngine
        sys.path.append('backend')
        from backend.real_prediction_engine import RealPredictionEngine
        
        engine = RealPredictionEngine()
        
        # Check if predict method exists
        if hasattr(engine, 'predict'):
            print("✅ RealPredictionEngine.predict(): Available")
        else:
            print("❌ RealPredictionEngine.predict(): Missing")
            
        # Check if set_data_collector method exists
        if hasattr(engine, 'set_data_collector'):
            print("✅ RealPredictionEngine.set_data_collector(): Available")
        else:
            print("❌ RealPredictionEngine.set_data_collector(): Missing")
            
        # Check RealDataCollector
        from backend.real_data_collector import RealDataCollector
        
        collector = RealDataCollector()
        
        if hasattr(collector, 'start_capture'):
            print("✅ RealDataCollector.start_capture(): Available")
        else:
            print("❌ RealDataCollector.start_capture(): Missing")
            
        if hasattr(collector, 'get_realtime_data'):
            print("✅ RealDataCollector.get_realtime_data(): Available")
        else:
            print("❌ RealDataCollector.get_realtime_data(): Missing")
            
    except Exception as e:
        print(f"❌ Backend check failed: {e}")

if __name__ == "__main__":
    print("🔍 Aviator Predictor - System Check")
    print("=" * 40)
    
    # Check dependencies
    check_dependencies()
    
    # Check file structure
    check_file_structure()
    
    # Check backend methods
    check_backend_methods()
    
    print("\n" + "=" * 40)
    print("✅ System check completed!")
    print("\nIf most items show ✅, your system is ready to run!")