from boto3.session import Session
from botocore.exceptions import ClientError
from pathlib import Path

class S3DataFetch:
    """[S3DataFetch]
    This class is responsible for downloading the data from S3 to a specified path in
    local directory
    """

    def __init__(self, bucket_name: str, file_name: str, save_path: Path):

        self.bucket_name = bucket_name
        self.file_name = file_name
        self.save_path = save_path
        self.session = Session()

    def initialize_resource(self):
        """[initialize_resource]
        This function requires the AWS Client credentials to be in the `default`
        profile in .aws/credentials file
        Change accordingly if needed
        """
        try:
            self.s3 = self.session.resource('s3')
        except ClientError as e:
            print(f"Error because: {e}")

    def get_file(self):

        """[get_file]
        This function requires the bucket name and the local save directore and will
        download from S3 using the boto3 session
        """

        self.initialize_resource()
        bucket = self.s3.Bucket(self.bucket_name)
        bucket.download_file(self.file_name, str(Path(self.save_path/self.file_name).absolute()))

#if __name__ == "__main__":

#    data_fetch = S3DataFetch(bucket_name="mlops-wine-demo", file_name="winequality-red.csv", save_path=Path.cwd().parents[1]/'data/raw')
#    data_fetch.get_file()