#!/home/nmrsu/miniconda/bin/python

import matplotlib.pyplot as pl
import matplotlib.cm as cm
import numpy as np
import scipy as sp
import sys
import os.path as osp
#print(sys.path)
#import montage as mt
#import phantom as ph
import brukerMRI.BrukerMRI as bm
import nibabel as nb
from scipy import ndimage
from scipy import stats
import time
from KalmanFilter import kf_predict, kf_update
#kalman from arxiv.org/pdf/1204.0375.pdf (implementation of kalman filter with python language, mohamed laaraiedh)


#scan_directory='/opt/PV6.0.1/data/nmrsu/20150910_094342_test1_1_236/'
#params=bm.ReadExperiment(scan_directory,12)

if len(sys.argv)<=2:
	print(str(sys.argv[0]),' <directory to watch>')
	print('e.g. /opt/PV6.0.1/data/nmrsu/20150910_094342_test1_1_236/12/pdata/1')
	sys.exit(0)

if len(sys.argv)<6:
	print('Defaulting to 1mm isotropic resolution')
	xres=1.0
	yres=1.0
	zres=1.0
	nreps=99999
	#entres=1.0
else:
	xres=float(sys.argv[2])
	yres=float(sys.argv[3])
	zres=float(sys.argv[4])
	nreps=float(sys.argv[5])
entres=2.0  #scaling it so the variations are on the order of the spatial ones
print('xres',xres,'yres',yres,'zres',zres,'entres',entres,'nreps',nreps)


scan_directory=str(sys.argv[1])
print('Watching ',scan_directory)

#/opt/PV6.0.1/data/nmrsu/20150910_094342_test1_1_236/12/pdata/1
reco=bm.ReadParamFile(osp.join(scan_directory,'reco'))


def ReadProcessedDataFrame(filename,reco,frame):
    sz=(reco["RECO_size"][0],reco["RECO_size"][1],reco["RecoObjectsPerRepetition"],reco["RecoNumRepetitions"])
    #print(sz)
    framesize=sz[0]*sz[1]*sz[2]
    targetframe=frame*framesize*np.dtype(np.int16).itemsize
    targetframeend=((frame+1)*framesize-1)*np.dtype(np.int16).itemsize #the last object in the desired frame
    #print(targetframe,targetframeend)
    with open(filename,"rb") as f:
        f.seek(0,2) #find the end of the file
        timeout = time.time() + 10   # 10 seconds from now
        while f.tell()<targetframeend and time.time() < timeout: #while the file is not written or 10 seconds elapses
            f.seek(0,2)
        if f.tell()>=targetframeend:
            #print('the frame is available')
            f.seek(targetframe) #move to the right place
            return np.fromfile(f,dtype=np.int16,count=framesize).reshape(sz[0:3],order="F")
        else:
            #print('frame is not available!')
            return np.array([])
        #if f.seek(framecount*frame):
        #    return np.fromfile(f,dtype=np.int16,count=framecount)


#initialize kalman state
dt = 0.7 #seconds, but as long as it's consistent it's OK
X_x = np.array([[0.0], [4.89300000e+02], [0.0], [1.0]])
P_x = np.array([[ 11.19946684,   0.        ,   0.94234035,   0.        ],
       [  0.        ,  11.19946684,   0.        ,   0.94234035],
       [  0.94234035,   0.        ,   0.16978195,   0.        ],
       [  0.        ,   0.94234035,   0.        ,   0.16978195]]) #initialized from the outputs of a scan
#P_x = np.diag((11.19946684, 11.19946684, 0.16978195, 0.16978195)) #what are good initializers?
A_x = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
Q_x = np.power(0.1,2)*np.eye(X_x.shape[0])  #noise in the system, smaller numbers are smoother
B_x = np.eye(X_x.shape[0])
U_x = np.zeros((X_x.shape[0],1))
X_y = np.array([[0.0], [4.89300000e+02], [0.0], [1.0]])
P_y = np.copy(P_x)
#P_y = np.diag((11.19946684, 11.19946684, 0.16978195, 0.16978195)) #what are good initializers?
A_y = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
Q_y = np.power(0.1,2)*np.eye(X_y.shape[0])  #noise in the system, smaller numbers are smoother
B_y = np.eye(X_y.shape[0])
U_y = np.zeros((X_y.shape[0],1))
X_z = np.array([[0.0], [4.89300000e+02], [0.0], [1.0]])
#P_z = np.diag((11.19946684, 11.19946684, 0.16978195, 0.16978195)) #what are good initializers?
P_z = np.copy(P_x)
A_z = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
Q_z = np.power(0.1,2)*np.eye(X_z.shape[0])  #noise in the system, smaller numbers are smoother
B_z = np.eye(X_z.shape[0])
U_z = np.zeros((X_z.shape[0],1))
X_e = np.array([[0.0], [4.89300000e+02], [0.0], [1.0]])
#P_e = np.diag((11.19946684, 11.19946684, 0.16978195, 0.16978195)) #what are good initializers?
P_e = np.copy(P_x)
A_e = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
Q_e = np.power(0.1,2)*np.eye(X_e.shape[0])  #noise in the system, smaller numbers are smoother
B_e = np.eye(X_e.shape[0])
U_e = np.zeros((X_e.shape[0],1))

#X_e = np.array([[0.0], [4.89300000e+02], [0.0], [1.0]])
#P_e = np.diag((0.01, 0.01, 0.01, 0.01)) #what are good initializers?
#A_e = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
#Q_e = np.power(0.1,2)*np.eye(X_e.shape[0])  #noise in the system, smaller numbers are smoother
#B_e = np.eye(X_e.shape[0])
#U_e = np.zeros((X_e.shape[0],1))

# initialize kalman measurement matrices 
Y_x = np.array([[X_x[0,0]], [X_x[1,0]]])
H_x = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
R_x = np.power(10.0,2)*np.eye(Y_x.shape[0])  #noise in the input, larger numbers are smoother
Y_y = np.array([[X_y[0,0]], [X_y[1,0]]])
H_y = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
R_y = np.power(10.0,2)*np.eye(Y_y.shape[0])  #noise in the input, larger numbers are smoother
Y_z = np.array([[X_z[0,0]], [X_z[1,0]]])
H_z = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
R_z = np.power(10.0,2)*np.eye(Y_z.shape[0])  #noise in the input, larger numbers are smoother
Y_e = np.array([[X_e[0,0]], [X_e[1,0]]])
H_e = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
R_e = np.power(10.0,2)*np.eye(Y_e.shape[0])  #noise in the input, larger numbers are smoother


#find the frame size so we know when we fall behind
sz=(reco["RECO_size"][0],reco["RECO_size"][1],reco["RecoObjectsPerRepetition"],reco["RecoNumRepetitions"])
framesize=sz[0]*sz[1]*sz[2]*np.dtype(np.int16).itemsize

#fig=pl.figure()
#ax=pl.axis()
fig,ax=pl.subplots(2,2,figsize=(20,12))
#print(ax)
#fig,ax1=pl.subplots(2,2)
pl.ion()
pl.show()
ax[0][0].legend(['x','y','z','e'],loc=3,framealpha=0.5,numpoints=1)
ax[0][0].set_xlabel('frame number')
ax[0][0].set_ylabel('position change (mm)')
ax[0][0].set_title('Real-time fMRI plotting')
ax[0][1].set_title('top')
ax[1][1].set_title('top')

#pl.show()
pl.draw()
pl.show()
background=fig.canvas.copy_from_bbox(ax[0][0].bbox)
points=ax[0][0].plot(0,0,'ro',0,'go',0,'bo',0,'co',0,'r-',0,'g-',0,'b-',0,'c-')

idx=0
a=ReadProcessedDataFrame(osp.join(scan_directory,'2dseq'),reco,idx)
#a=ReadProcessedDataFrame('/opt/PV6.0.1/data/nmrsu/20150911_094342_test1_1_236/12/pdata/1/2dseq',reco,idx)
pos=ndimage.measurements.center_of_mass(a)
ent=stats.entropy(a.ravel())
xoff=pos[0]
yoff=pos[1]
zoff=pos[2]
eoff=ent
xar=[]
yar=[]
zar=[]
entar=[]
kxar=[]
kyar=[]
kzar=[]
kentar=[]
dmin=0
dmax=0
t=0
vlines=[]

sz=a.shape
midx=np.round(xoff) #sz[0]//2
midy=np.round(yoff) #sz[1]//2
midz=np.round(zoff) #sz[2]//2

im1=ax[0][1].imshow(np.rot90(a[:,:,midz].squeeze(),1),cmap=cm.Greys_r)
im2=ax[1][0].imshow(np.rot90(a[:,midy,:].squeeze(),0),cmap=cm.Greys_r)
ax[1][0].set_aspect(xres/zres) #x or y?
im3=ax[1][1].imshow(np.rot90(a[midx,:,:].squeeze(),2),cmap=cm.Greys_r)
ax[1][1].set_aspect(yres/zres) #x or y?

while a.size>0:
    pos=ndimage.measurements.center_of_mass(a)
    ent=stats.entropy(a.ravel())
    print(idx,pos,ent)
    #deal with kalman
    (X_x, P_x) = kf_predict(X_x, P_x, A_x, Q_x, B_x, U_x) 
    Y_x = np.array([xres*([pos[0]]-xoff),[t]])
    (X_x, P_x, K_x, IM_x, IS_x, LH_x) = kf_update(X_x, P_x, Y_x, H_x, R_x) 
    kxar.append(X_x[0,0])
    (X_y, P_y) = kf_predict(X_y, P_y, A_y, Q_y, B_y, U_y) 
    Y_y = np.array([yres*([pos[1]]-yoff),[t]])
    (X_y, P_y, K_y, IM_y, IS_y, LH_y) = kf_update(X_y, P_y, Y_y, H_y, R_y) 
    kyar.append(X_y[0,0])
    #print('y',Y_y,'x',X_y,'p',P_y,'k',K_y,'im',IM_y,'is',IS_y,'lh',LH_y)
    (X_z, P_z) = kf_predict(X_z, P_z, A_z, Q_z, B_z, U_z) 
    Y_z = np.array([zres*([pos[2]]-zoff),[t]])
    (X_z, P_z, K_z, IM_z, IS_z, LH_z) = kf_update(X_z, P_z, Y_z, H_z, R_z) 
    kzar.append(X_z[0,0])
    (X_e, P_e) = kf_predict(X_e, P_e, A_e, Q_e, B_e, U_e) 
    Y_e = np.array([entres*([ent]-eoff),[t]])
    (X_e, P_e, K_e, IM_e, IS_e, LH_e) = kf_update(X_e, P_e, Y_e, H_e, R_e) 
    kentar.append(X_e[0,0])

    #is it OK? let's try a magnitude and velocity approach
    mag_dx=np.abs(Y_x[0,0]-X_x[0,0])
    mag_dy=np.abs(Y_y[0,0]-X_y[0,0])
    mag_dz=np.abs(Y_z[0,0]-X_z[0,0])
    mag_de=np.abs(Y_e[0,0]-X_e[0,0])
    mag_vx=np.abs(X_x[2,0])
    mag_vy=np.abs(X_y[2,0])
    mag_vz=np.abs(X_z[2,0])
    mag_ve=np.abs(X_e[2,0])
    #print('x',mag_dx,'y',mag_dy,'z',mag_dz,'e',mag_de,'xv',mag_vx,'yv',mag_vy,'zv',mag_vz,'ev',mag_ve)
    mot=np.linalg.norm(np.array([mag_dx,mag_dy,mag_dz,mag_de,3*mag_vx,3*mag_vy,3*mag_vz,3*mag_ve]))  #up gain the velocity a little
    #print(mot)
    #print(X_x)

    #plotting
    xar.append(xres*(pos[0]-xoff))
    yar.append(yres*(pos[1]-yoff))
    zar.append(zres*(pos[2]-zoff))
    entar.append(entres*(ent-eoff))
    idxar=range(len(xar))
    dmin=min(xres*(pos[0]-xoff),yres*(pos[1]-yoff),zres*(pos[2]-zoff),entres*(ent-eoff),dmin)
    dmax=max(xres*(pos[0]-xoff),yres*(pos[1]-yoff),zres*(pos[2]-zoff),entres*(ent-eoff),dmax)
    points[0].set_data(idxar,xar)
    points[1].set_data(idxar,yar)
    points[2].set_data(idxar,zar)
    points[3].set_data(idxar,entar)
    points[4].set_data(idxar,kxar)
    points[5].set_data(idxar,kyar)
    points[6].set_data(idxar,kzar)
    points[7].set_data(idxar,kentar)
    fig.canvas.restore_region(background)
    ax[0][0].set_xlim([0,len(xar)])
    ax[0][0].set_ylim([dmin,dmax])
    ax[0][0].draw_artist(points[0])
    ax[0][0].draw_artist(points[1])
    ax[0][0].draw_artist(points[2])
    ax[0][0].draw_artist(points[3])
    ax[0][0].draw_artist(points[4])
    ax[0][0].draw_artist(points[5])
    ax[0][0].draw_artist(points[6])
    ax[0][0].draw_artist(points[7])
    ax[0][0].legend(['x','y','z','e'],loc=3,framealpha=0.5,numpoints=1)

    if mot>0.02:
        #print('motihon!!!!')
	print(mot)
        vlines.append(idx)
        ax[0][0].vlines(vlines,dmin,dmax)
    #print(mot)
    fig.canvas.blit(ax[0][0].bbox)

    
    im1.set_data(np.rot90(a[:,:,midz].squeeze(),1))
    im2.set_data(np.rot90(a[:,midy,:].squeeze(),0))
    im3.set_data(np.rot90(a[midx,:,:].squeeze(),2))
    fig.canvas.draw()
    #pl.draw()
    idx+=1
    t+=dt
    if idx<nreps:
    	a=ReadProcessedDataFrame(osp.join(scan_directory,'2dseq'),reco,idx)
    else:
	a=np.array([])
    

extent = ax[0][0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('/tmp/realtime_motion_plot.png', bbox_inches=extent.expanded(1.25, 1.25))
print((X_x, P_x, K_x, IM_x, IS_x, LH_x))
print((X_y, P_y, K_y, IM_y, IS_y, LH_y))
print((X_z, P_z, K_z, IM_z, IS_z, LH_z))
print((X_e, P_e, K_e, IM_e, IS_e, LH_e))
pl.show(block=True)

