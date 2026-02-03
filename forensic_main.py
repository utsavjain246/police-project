
import argparse
from recorded_orchestrator import RecordedOrchestrator

def main():
    parser = argparse.ArgumentParser(description="Forensic Video Analysis")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--query", required=True, help="Natural language query (e.g. 'Find red car')")
    
    args = parser.parse_args()
    
    orchestrator = RecordedOrchestrator()
    report = orchestrator.process(args.video_path, args.query)
    
    print("\nanalysis Complete.")
    print(f"Found {len(report['results'])} events.")
    print("Report saved to forensic_report.json")

if __name__ == "__main__":
    main()
