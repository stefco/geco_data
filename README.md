# GECo Data

Scripts for manipulating timing data stored in GW frame files.

## Usage

So far, using these should just involve placing them in a folder in your path
and then calling the commands by name. Most of them take arguments involving
start/end times and possibly channel names. So far, for simplicity's sake,
they all assume that you are keeping your work in the same directory. They
are simple scripts for doing simple things reliably with large amounts of
data.

## Tips

After dumping all that data. you should probably zip up your output directories
to make it easier to transfer the data elsewhere.  Compressing the data will
speed up file transfers, but more importantly, representing all of the files in
your output directories as a single zipped archive will make file transfer much
faster (downloading tens of thousands of small files will take many, many times
longer than downloading a single large file, even if the total amount of data
is nominally the same). You can do this like so:

```bash
zip -r output_file_name.zip input_directory
```

You can then do something clever like, say, uploading your data to a remote
server using `scp` or something like the pure-`bash`
[Dropbox-Uploader](https://github.com/andreafabrizi/Dropbox-Uploader) script.
