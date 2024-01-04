import torch
import cv2
import numpy as np
def RadiusNorm(n, org=None):
    """
    Create an array where each value is the normalized distance from the origin.
    The array size will be n x n.
    """
    #if not isinstance(n, tuple):
    #    n = (n, n)
    
    if org is None:
        org = torch.ceil((n + 1) / 2).int()

    x, y = org
    
    Y, X = torch.meshgrid([torch.arange(1-x, n[0]-x+1,device=n.device), torch.arange(1-y, n[1]-y+1,device=n.device)])
    Y = Y.to(torch.float32) / n[0]
    X = X.to(torch.float32) / n[1]
    r = torch.sqrt(X**2 + Y**2)
    theta = torch.atan2(Y, X)

    return r, theta

def cryo_CTF_Relion(n, voltage, DefocusU, DefocusV, DefocusAngle,
                    SphericalAberration, pixA, AmplitudeContrast):
    """
    Compute the contrast transfer function (CTF) for an electron microscope image.
    """
    # Constants and conversions
    lambda_val = 1.22639 / torch.sqrt(voltage * 1000 + 0.97845 * voltage**2)
    BW = 1 / (pixA / 10)

    # Calculate radii and angles
    s, theta = RadiusNorm(n)
    s = s * BW

    DFavg = (DefocusU + DefocusV) / 2
    DFdiff = (DefocusU - DefocusV)
    df = DFavg + DFdiff * torch.cos(2 * (theta - DefocusAngle)) / 2

    k2 = torch.pi * lambda_val * df
    k4 = torch.pi / 2 * 1e6 * SphericalAberration * lambda_val**3
    chi = k4 * s**4 - k2 * s**2
    h = torch.sqrt(1 - AmplitudeContrast**2) * torch.sin(chi) - AmplitudeContrast * torch.cos(chi)

    return h

def phase_flip(im, n, voltage, DefocusU, DefocusV, DefocusAngle, Cs, pixA, A):
    """
    Perform phase flipping on an image using the CTF.
    """
    h = cryo_CTF_Relion(n, voltage, DefocusU, DefocusV, DefocusAngle, Cs, pixA, A)

    # Phase flip
    imhat = torch.fft.fftshift(torch.fft.fft2(im))
    pfim = torch.fft.ifft2(torch.fft.ifftshift(imhat * torch.sign(h)))

    # Check for large imaginary components
    if n[0] % 2 == 1:
        imag_norm = torch.linalg.norm(pfim.imag)
        total_norm = torch.linalg.norm(pfim)
        if imag_norm / total_norm > 5.0e-7:
            print(f'Warning: Large imaginary components in image = {imag_norm / total_norm}')

    return pfim.real.type(torch.float32)


if __name__ == '__main__':
    device='cuda'
    n = torch.tensor((4096,4096)).to(device)
    voltage = torch.tensor(300).to(device)  # In kV
    DefocusU = torch.tensor(3036.466 ).to(device) # In nm
    DefocusV = torch.tensor(3062.678).to(device)  # In nm
    DefocusAngle = torch.tensor(np.radians(41.90)).to(device)  # Convert degrees to radians
    Cs = torch.tensor(2.0).to(device)  # In mm
    pixA = torch.tensor(1.34).to(device)  # In Angstrom
    AmplitudeContrast = torch.tensor(0.07).to(device)
    import mrcfile
    with mrcfile.mmap('../001_movie.mrcs',mode='r') as mrc:
        image_ = mrc.data/16383
    image = torch.from_numpy(image_).to(device)
    #image = np.mean(image, axis=0)
    output = []
    for i in range(16):
        pfim = phase_flip(image[i], n, voltage, DefocusU, DefocusV, DefocusAngle, Cs, pixA, AmplitudeContrast)**2
        pfim_cpu = pfim.cpu().numpy()
        output.append(pfim_cpu)
        cv2.imwrite('per_frame_torch_frame_{frame}.png'.format(frame=i), (pfim_cpu*255*20).astype(np.uint8))
        cv2.imwrite('per_frame_torch_frame_{frame}_non_fix.png'.format(frame=i), (image_[i]*255*20).astype(np.uint8))
    cv2.imwrite('per_frame_torch.png', (np.mean(output, axis=0)*255*10).astype(np.uint8))
