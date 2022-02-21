# Difference hashing


def dhash(image, hash_size=8):
    	"""
	Difference Hash computation.
	following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
	computes differences horizontally
	@image must be a PIL instance.
	"""
	# resize(w, h), but numpy.array((h, w))
	if hash_size < 2:
		raise ValueError("Hash size must be greater than or equal to 2")

	image = image.convert("L").resize((hash_size + 1, hash_size), Image.ANTIALIAS)
	pixels = numpy.asarray(image)
	# compute differences between columns
	diff = pixels[:, 1:] > pixels[:, :-1]
	return ImageHash(diff)

def dhash_vertical(image, hash_size=8):
    	"""
	Difference Hash computation.
	following http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
	computes differences vertically
	@image must be a PIL instance.
	"""
	# resize(w, h), but numpy.array((h, w))
	image = image.convert("L").resize((hash_size, hash_size + 1), Image.ANTIALIAS)
	pixels = numpy.asarray(image)
	# compute differences between rows
	diff = pixels[1:, :] > pixels[:-1, :]
	return ImageHash(diff)