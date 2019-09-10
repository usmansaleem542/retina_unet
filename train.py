from Models import Models
import generator as gen

data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05,
                    height_shift_range=0.05, shear_range=0.05,
                    zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')

path = 'data/retina/'
myGene = gen.trainGenerator(2, path+'train','image','label',data_gen_args,save_to_dir = None)
model = Models(input_size = (256,256,1), model='unet', modelPath='unet_retina.hdf5')
model.train(myGene, steps=3, epochs=1)
model.predict_images('data/retina/test/')
